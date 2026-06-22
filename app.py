"""
Production-Ready AI Agent Web Application
- Web Search | Calculator | Document Query
- Full Document Management with Upload/Delete/Clear
- Professional UI with File Management Panel
"""

import os
import json
import math
import re
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
import PyPDF2
import docx
import csv
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import shutil

# ==================== Configuration ====================

UPLOAD_FOLDER = 'uploads'
DOCUMENT_STORE_FOLDER = 'document_store'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'md', 'py', 'js', 'html', 'css', 'json', 'xml'}

# Create folders
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(DOCUMENT_STORE_FOLDER).mkdir(exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==================== Backend Components ====================

class DocumentStore:
    """Production-ready document store with embeddings"""
    
    def __init__(self, storage_path: str = DOCUMENT_STORE_FOLDER):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.documents = []
        self.embeddings = []
        self.model = None
        self.load_model()
        self.load_store()
    
    def load_model(self):
        """Load sentence transformer model"""
        try:
            print("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        try:
            if file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_path.suffix.lower() == '.pdf':
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text if text.strip() else "No extractable text found in PDF."
            
            elif file_path.suffix.lower() == '.docx':
                doc = docx.Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
                return text if text.strip() else "No text found in document."
            
            elif file_path.suffix.lower() == '.csv':
                text = ""
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        text += ' '.join(row) + '\n'
                return text
            
            else:
                # Try as text for unknown formats
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def add_document(self, file_path: str, metadata: Optional[Dict] = None) -> Dict:
        """Add a document to the store"""
        if self.model is None:
            return {"success": False, "message": "Model not loaded. Cannot add document."}
        
        text = self.extract_text_from_file(file_path)
        if text.startswith("ERROR:"):
            return {"success": False, "message": text}
        
        if not text.strip():
            return {"success": False, "message": "Document is empty or no text could be extracted."}
        
        # Remove existing document with same name
        filename = Path(file_path).name
        self.remove_document(filename, save=False)
        
        chunks = self.chunk_text(text, chunk_size=500, overlap=50)
        
        for i, chunk in enumerate(chunks):
            doc_entry = {
                'file_path': str(file_path),
                'file_name': filename,
                'file_size': os.path.getsize(file_path),
                'file_type': Path(file_path).suffix,
                'chunk_index': i,
                'text': chunk,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            self.documents.append(doc_entry)
            embedding = self.model.encode(chunk)
            self.embeddings.append(embedding)
        
        self.save_store()
        return {
            "success": True,
            "message": f"✅ Added '{filename}' ({len(chunks)} chunks, {len(text)} characters)",
            "chunks": len(chunks),
            "filename": filename,
            "size": len(text),
            "file_type": Path(file_path).suffix
        }
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [text]
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search documents using semantic similarity"""
        if self.model is None or len(self.documents) == 0:
            return []
        
        query_embedding = self.model.encode(query)
        
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.2:
                result = self.documents[idx].copy()
                result['similarity'] = float(similarities[idx])
                results.append(result)
        
        return results
    
    def save_store(self):
        """Save document store to disk"""
        store_data = {
            'documents': self.documents,
            'embeddings': [emb.tolist() for emb in self.embeddings]
        }
        store_file = self.storage_path / 'store.pkl'
        with open(store_file, 'wb') as f:
            pickle.dump(store_data, f)
    
    def load_store(self):
        """Load document store from disk"""
        store_file = self.storage_path / 'store.pkl'
        if store_file.exists():
            try:
                with open(store_file, 'rb') as f:
                    store_data = pickle.load(f)
                    self.documents = store_data['documents']
                    self.embeddings = [np.array(emb) for emb in store_data['embeddings']]
                print(f"✅ Loaded {len(self.documents)} document chunks from storage")
            except Exception as e:
                print(f"⚠️ Error loading store: {e}")
                self.documents = []
                self.embeddings = []
    
    def get_stats(self) -> Dict:
        """Get store statistics"""
        unique_files = {}
        for doc in self.documents:
            filename = doc['file_name']
            if filename not in unique_files:
                unique_files[filename] = {
                    'name': filename,
                    'type': doc.get('file_type', 'unknown'),
                    'size': doc.get('file_size', 0),
                    'chunks': 0,
                    'added_date': doc.get('timestamp', 'unknown')
                }
            unique_files[filename]['chunks'] += 1
        
        file_list = list(unique_files.values())
        file_types = list(set(doc.get('file_type', 'unknown') for doc in self.documents))
        
        return {
            'total_documents': len(file_list),
            'total_chunks': len(self.documents),
            'total_size_chars': sum(len(doc['text']) for doc in self.documents),
            'file_types': file_types,
            'files': file_list,
            'total_size_bytes': sum(doc.get('file_size', 0) for doc in self.documents if 'file_size' in doc)
        }
    
    def list_documents(self) -> List[Dict]:
        """List all unique documents with details"""
        unique_files = {}
        for doc in self.documents:
            filename = doc['file_name']
            if filename not in unique_files:
                unique_files[filename] = {
                    'name': filename,
                    'type': doc.get('file_type', 'unknown'),
                    'size': doc.get('file_size', 0),
                    'chunks': 0,
                    'added_date': doc.get('timestamp', 'unknown')
                }
            unique_files[filename]['chunks'] += 1
        
        return list(unique_files.values())
    
    def remove_document(self, filename: str, save: bool = True) -> Dict:
        """Remove a document from the store"""
        initial_count = len(self.documents)
        
        indices_to_remove = [
            i for i, doc in enumerate(self.documents) 
            if doc['file_name'] == filename
        ]
        
        if not indices_to_remove:
            return {"success": False, "message": f"Document '{filename}' not found."}
        
        for idx in reversed(indices_to_remove):
            self.documents.pop(idx)
            self.embeddings.pop(idx)
        
        if save:
            self.save_store()
        
        removed_count = initial_count - len(self.documents)
        return {
            "success": True,
            "message": f"✅ Removed '{filename}' ({removed_count} chunks)",
            "removed_chunks": removed_count
        }
    
    def clear_all(self) -> Dict:
        """Clear all documents from store"""
        count = len(self.documents)
        self.documents = []
        self.embeddings = []
        self.save_store()
        
        # Also clear upload folder
        upload_path = Path(UPLOAD_FOLDER)
        if upload_path.exists():
            for file in upload_path.iterdir():
                if file.is_file():
                    file.unlink()
        
        return {
            "success": True,
            "message": f"✅ Cleared all documents ({count} chunks removed)",
            "removed_chunks": count
        }


class WebSearcher:
    """Web search using DuckDuckGo"""
    
    def __init__(self):
        self.ddgs = DDGS()
    
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Perform web search"""
        try:
            results = []
            search_results = self.ddgs.text(query, max_results=max_results)
            
            for result in search_results:
                results.append({
                    'title': result.get('title', 'No title'),
                    'link': result.get('link', '#'),
                    'snippet': result.get('body', 'No description available')[:300]
                })
            
            return results if results else [{'error': 'No results found'}]
        except Exception as e:
            return [{'error': f"Search failed: {str(e)}"}]
    
    def search_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search news articles"""
        try:
            results = []
            news_results = self.ddgs.news(query, max_results=max_results)
            
            for result in news_results:
                results.append({
                    'title': result.get('title', 'No title'),
                    'link': result.get('url', '#'),
                    'snippet': result.get('body', 'No description')[:300],
                    'date': result.get('date', 'N/A'),
                    'source': result.get('source', 'Unknown')
                })
            
            return results if results else [{'error': 'No news found'}]
        except Exception as e:
            return [{'error': f"News search failed: {str(e)}"}]


class Calculator:
    """Advanced calculator"""
    
    def calculate(self, expression: str) -> Dict:
        """Evaluate mathematical expressions safely"""
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow, 'sqrt': math.sqrt,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'log': math.log, 'log10': math.log10, 'log2': math.log2,
            'exp': math.exp, 'pi': math.pi, 'e': math.e,
            'factorial': math.factorial, 'gcd': math.gcd,
            'degrees': math.degrees, 'radians': math.radians
        }
        
        try:
            expression = expression.strip()
            original = expression
            expression = expression.replace('^', '**')
            expression = expression.replace('×', '*')
            expression = expression.replace('÷', '/')
            expression = expression.replace('π', 'pi')
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            
            if isinstance(result, float):
                if result == int(result):
                    result = int(result)
                else:
                    result = round(result, 10)
            
            return {
                "success": True,
                "expression": original,
                "result": str(result)
            }
        except Exception as e:
            return {
                "success": False,
                "expression": expression,
                "error": str(e)
            }
    
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> Dict:
        """Convert between common units"""
        length_to_meters = {
            'mm': 0.001, 'cm': 0.01, 'm': 1, 'km': 1000,
            'in': 0.0254, 'inch': 0.0254, 'ft': 0.3048, 'feet': 0.3048,
            'yd': 0.9144, 'yard': 0.9144, 'mi': 1609.34, 'mile': 1609.34, 'miles': 1609.34
        }
        
        weight_to_grams = {
            'mg': 0.001, 'g': 1, 'gram': 1, 'kg': 1000, 'kilogram': 1000,
            'oz': 28.3495, 'ounce': 28.3495, 'lb': 453.592, 'pound': 453.592,
            'lbs': 453.592, 'pounds': 453.592
        }
        
        try:
            # Temperature
            temp_units = ['c', 'celsius', 'f', 'fahrenheit', 'k', 'kelvin']
            if from_unit.lower() in temp_units and to_unit.lower() in temp_units:
                result = self._convert_temperature(value, from_unit, to_unit)
                return {"success": True, "result": result}
            
            # Length
            if from_unit.lower() in length_to_meters and to_unit.lower() in length_to_meters:
                meters = value * length_to_meters[from_unit.lower()]
                result = meters / length_to_meters[to_unit.lower()]
                return {"success": True, "result": f"{value} {from_unit} = {result:.4f} {to_unit}"}
            
            # Weight
            if from_unit.lower() in weight_to_grams and to_unit.lower() in weight_to_grams:
                grams = value * weight_to_grams[from_unit.lower()]
                result = grams / weight_to_grams[to_unit.lower()]
                return {"success": True, "result": f"{value} {from_unit} = {result:.4f} {to_unit}"}
            
            return {"success": False, "error": "Unsupported unit conversion"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> str:
        from_unit = from_unit.upper()[0]
        to_unit = to_unit.upper()[0]
        
        if from_unit == 'F':
            celsius = (value - 32) * 5/9
        elif from_unit == 'K':
            celsius = value - 273.15
        else:
            celsius = value
        
        if to_unit == 'F':
            result = celsius * 9/5 + 32
        elif to_unit == 'K':
            result = celsius + 273.15
        else:
            result = celsius
        
        return f"{value}°{from_unit} = {result:.2f}°{to_unit}"


class AIAgent:
    """Main AI Agent"""
    
    def __init__(self):
        print("Initializing components...")
        self.web_searcher = WebSearcher()
        self.calculator = Calculator()
        self.document_store = DocumentStore()
        print("✅ AI Agent ready!")
    
    def process_command(self, user_input: str) -> Dict:
        """Process user input and return structured response"""
        user_input = user_input.strip()
        
        if not user_input:
            return {'type': 'error', 'message': 'Please enter a command or question.'}
        
        parts = user_input.split(' ', 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ''
        
        # Route commands
        routes = {
            'search': lambda: self._handle_search(args),
            'news': lambda: self._handle_news(args),
            'calc': lambda: self._handle_calc(args or user_input),
            'calculate': lambda: self._handle_calc(args or user_input),
            'math': lambda: self._handle_calc(args or user_input),
            'convert': lambda: self._handle_convert(args),
            'add_doc': lambda: self._handle_add_document(args),
            'add': lambda: self._handle_add_document(args),
            'upload': lambda: self._handle_add_document(args),
            'query_doc': lambda: self._handle_query_document(args),
            'find': lambda: self._handle_query_document(args),
            'search_doc': lambda: self._handle_query_document(args),
            'doc_stats': lambda: self._handle_doc_stats(),
            'stats': lambda: self._handle_doc_stats(),
            'documents': lambda: self._handle_doc_stats(),
            'list_docs': lambda: self._handle_list_documents(),
            'files': lambda: self._handle_list_documents(),
            'remove_doc': lambda: self._handle_remove_document(args),
            'delete': lambda: self._handle_remove_document(args),
            'clear_all': lambda: self._handle_clear_all(),
            'clear': lambda: self._handle_clear_all(),
            'help': lambda: self._handle_help(),
        }
        
        if command in routes:
            return routes[command]()
        
        # Auto-detect
        if self._is_math_expression(user_input):
            return self._handle_calc(user_input)
        
        if self._is_conversion(user_input):
            return self._handle_convert(user_input)
        
        # Default to web search
        return self._handle_search(user_input)
    
    def _handle_search(self, query: str) -> Dict:
        if not query:
            return {'type': 'error', 'message': 'What would you like to search for?'}
        
        results = self.web_searcher.search(query)
        
        if not results or ('error' in results[0]):
            return {'type': 'error', 'message': results[0].get('error', 'No results found')}
        
        return {'type': 'search', 'query': query, 'results': results, 'count': len(results)}
    
    def _handle_news(self, query: str) -> Dict:
        if not query:
            return {'type': 'error', 'message': 'What news topic?'}
        
        results = self.web_searcher.search_news(query)
        
        if not results or ('error' in results[0]):
            return {'type': 'error', 'message': results[0].get('error', 'No news found')}
        
        return {'type': 'news', 'query': query, 'results': results, 'count': len(results)}
    
    def _handle_calc(self, expression: str) -> Dict:
        if not expression:
            return {'type': 'error', 'message': 'What would you like to calculate?'}
        
        result = self.calculator.calculate(expression)
        
        if result['success']:
            return {'type': 'calc', 'expression': result['expression'], 'result': result['result']}
        else:
            return {'type': 'error', 'message': f"Calculation error: {result['error']}"}
    
    def _handle_convert(self, args: str) -> Dict:
        if not args:
            return {'type': 'error', 'message': 'Format: convert <value> <from_unit> to <to_unit>'}
        
        try:
            text = args.lower().replace('convert', '').strip()
            parts = text.split(' to ')
            if len(parts) != 2:
                return {'type': 'error', 'message': 'Format: convert <value> <from_unit> to <to_unit>'}
            
            from_part = parts[0].strip().split()
            value = float(from_part[0])
            from_unit = from_part[1]
            to_unit = parts[1].strip()
            
            result = self.calculator.convert_units(value, from_unit, to_unit)
            
            if result['success']:
                return {'type': 'convert', 'result': result['result']}
            else:
                return {'type': 'error', 'message': result['error']}
        except:
            return {'type': 'error', 'message': 'Invalid format. Example: convert 100 km to miles'}
    
    def _handle_add_document(self, file_path: str) -> Dict:
        if not file_path:
            return {'type': 'error', 'message': 'Please provide a file path or upload a file.'}
        
        file_path = file_path.strip().strip('"\'')
        if not os.path.exists(file_path):
            return {'type': 'error', 'message': f'❌ File not found: {file_path}'}
        
        if not allowed_file(file_path):
            return {'type': 'error', 'message': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}
        
        result = self.document_store.add_document(file_path)
        
        if result['success']:
            return {
                'type': 'document',
                'message': result['message'],
                'filename': result['filename'],
                'chunks': result['chunks']
            }
        else:
            return {'type': 'error', 'message': result['message']}
    
    def _handle_query_document(self, query: str) -> Dict:
        if not query:
            return {'type': 'error', 'message': 'What would you like to find in your documents?'}
        
        results = self.document_store.search(query)
        
        if not results:
            return {'type': 'document_search', 'query': query, 'results': [], 'count': 0}
        
        return {'type': 'document_search', 'query': query, 'results': results[:5], 'count': len(results)}
    
    def _handle_doc_stats(self) -> Dict:
        stats = self.document_store.get_stats()
        return {'type': 'document_stats', 'stats': stats}
    
    def _handle_list_documents(self) -> Dict:
        docs = self.document_store.list_documents()
        return {'type': 'document_list', 'files': docs, 'count': len(docs)}
    
    def _handle_remove_document(self, filename: str) -> Dict:
        if not filename:
            return {'type': 'error', 'message': 'Please specify a filename to remove.'}
        
        result = self.document_store.remove_document(filename)
        
        if result['success']:
            return {'type': 'document', 'message': result['message']}
        else:
            return {'type': 'error', 'message': result['message']}
    
    def _handle_clear_all(self) -> Dict:
        result = self.document_store.clear_all()
        return {'type': 'document', 'message': result['message']}
    
    def _handle_help(self) -> Dict:
        return {
            'type': 'help',
            'commands': [
                {'cmd': 'search <query>', 'desc': 'Search the web', 'example': 'search latest AI news'},
                {'cmd': 'news <topic>', 'desc': 'Latest news', 'example': 'news technology'},
                {'cmd': 'calc <expression>', 'desc': 'Calculate', 'example': 'calc 15 * 3 + 27'},
                {'cmd': 'convert <value> <from> to <to>', 'desc': 'Convert units', 'example': 'convert 100 km to miles'},
                {'cmd': 'add_doc <path>', 'desc': 'Add document', 'example': 'add_doc report.txt'},
                {'cmd': 'query_doc <text>', 'desc': 'Search documents', 'example': 'query_doc AI trends'},
                {'cmd': 'doc_stats', 'desc': 'Document statistics', 'example': 'doc_stats'},
                {'cmd': 'list_docs', 'desc': 'List all documents', 'example': 'list_docs'},
                {'cmd': 'remove_doc <name>', 'desc': 'Remove document', 'example': 'remove_doc report.txt'},
                {'cmd': 'clear_all', 'desc': 'Clear all documents', 'example': 'clear_all'},
            ]
        }
    
    def _is_math_expression(self, text: str) -> bool:
        if len(text.split()) > 5:
            return False
        
        common_words = ['what', 'how', 'when', 'where', 'who', 'why', 'is', 'are', 'the', 'a', 'an', 'tell', 'show', 'find', 'get', 'search', 'news', 'help']
        if text.split()[0].lower() in common_words:
            return False
        
        math_functions = ['sqrt', 'sin', 'cos', 'tan', 'log', 'abs', 'pi', 'e', 'factorial']
        has_numbers = any(c.isdigit() for c in text)
        has_operators = any(op in text for op in ['+', '-', '*', '/', '^', '**'])
        has_functions = any(func in text.lower() for func in math_functions)
        
        return (has_numbers and has_operators) or has_functions
    
    def _is_conversion(self, text: str) -> bool:
        conversion_indicators = [' to ', 'convert']
        units = ['km', 'miles', 'kg', 'pounds', 'celsius', 'fahrenheit', 'cm', 'mm', 'ft', 'inch']
        
        has_conversion_word = any(indicator in text.lower() for indicator in conversion_indicators)
        has_units = any(unit in text.lower() for unit in units)
        
        return has_conversion_word and has_units


# ==================== Flask Web App ====================

app = Flask(__name__)
app.secret_key = 'production-ai-agent-2024'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

print("\n" + "="*60)
print("🤖 Initializing Production AI Agent...")
print("="*60)
agent = AIAgent()

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 AI Agent Pro - Web Search | Calculator | Documents</title>
    <style>
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #48bb78;
            --warning: #ed8936;
            --danger: #fc8181;
            --bg: #f7fafc;
            --card: #ffffff;
            --text: #2d3748;
            --border: #e2e8f0;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .app-container {
            width: 100%;
            max-width: 1400px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 25px 80px rgba(0,0,0,0.3);
            overflow: hidden;
            display: flex;
            height: 90vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: 320px;
            background: #2d3748;
            color: white;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #4a5568;
        }
        
        .sidebar-header {
            padding: 20px;
            background: #1a202c;
            text-align: center;
            border-bottom: 1px solid #4a5568;
        }
        
        .sidebar-header h2 {
            font-size: 20px;
            margin-bottom: 5px;
        }
        
        .sidebar-header p {
            font-size: 12px;
            opacity: 0.7;
        }
        
        .upload-section {
            padding: 20px;
            border-bottom: 1px solid #4a5568;
        }
        
        .upload-area {
            border: 2px dashed #4a5568;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 10px;
        }
        
        .upload-area:hover {
            border-color: var(--primary);
            background: rgba(102, 126, 234, 0.1);
        }
        
        .upload-area.dragover {
            border-color: var(--success);
            background: rgba(72, 187, 120, 0.1);
        }
        
        .upload-icon {
            font-size: 30px;
            margin-bottom: 10px;
        }
        
        .upload-text {
            font-size: 13px;
            color: #a0aec0;
        }
        
        #fileInput {
            display: none;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.3s;
            width: 100%;
            margin-bottom: 8px;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
        }
        
        .btn-primary:hover {
            background: #5a6fd6;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: var(--danger);
            color: white;
        }
        
        .btn-danger:hover {
            background: #f56565;
            transform: translateY(-2px);
        }
        
        .btn-success {
            background: var(--success);
            color: white;
        }
        
        .btn-success:hover {
            background: #38a169;
            transform: translateY(-2px);
        }
        
        .document-list {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
        }
        
        .document-list h3 {
            font-size: 14px;
            color: #a0aec0;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .doc-item {
            background: #4a5568;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            transition: all 0.3s;
        }
        
        .doc-item:hover {
            background: #5a6578;
            transform: translateX(5px);
        }
        
        .doc-item-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 8px;
        }
        
        .doc-name {
            font-size: 13px;
            font-weight: 600;
            word-break: break-word;
        }
        
        .doc-type {
            font-size: 10px;
            background: var(--primary);
            padding: 2px 8px;
            border-radius: 10px;
            white-space: nowrap;
        }
        
        .doc-meta {
            font-size: 11px;
            color: #a0aec0;
            display: flex;
            gap: 15px;
        }
        
        .remove-doc-btn {
            background: none;
            border: none;
            color: #fc8181;
            cursor: pointer;
            font-size: 16px;
            padding: 2px 5px;
            transition: all 0.3s;
        }
        
        .remove-doc-btn:hover {
            color: #f56565;
            transform: scale(1.2);
        }
        
        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #a0aec0;
        }
        
        .empty-state .icon {
            font-size: 50px;
            margin-bottom: 15px;
        }
        
        /* Main Chat Area */
        .main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        
        .main-header {
            padding: 15px 25px;
            background: white;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .main-header h1 {
            font-size: 22px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .quick-actions {
            display: flex;
            gap: 8px;
            padding: 10px 25px;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
        }
        
        .quick-btn {
            padding: 6px 14px;
            background: white;
            border: 1px solid var(--border);
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.3s;
            white-space: nowrap;
        }
        
        .quick-btn:hover {
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }
        
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 25px;
            background: var(--bg);
        }
        
        .message {
            margin-bottom: 20px;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            display: flex;
            justify-content: flex-end;
        }
        
        .user-bubble {
            background: var(--primary);
            color: white;
            padding: 12px 18px;
            border-radius: 18px 18px 5px 18px;
            max-width: 60%;
            word-wrap: break-word;
            font-size: 14px;
        }
        
        .agent-message {
            display: flex;
            justify-content: flex-start;
        }
        
        .agent-bubble {
            background: white;
            padding: 15px 20px;
            border-radius: 18px 18px 18px 5px;
            max-width: 75%;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            word-wrap: break-word;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .search-result, .news-result, .doc-result {
            margin: 10px 0;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .search-result { background: #f0f4ff; border-color: var(--primary); }
        .news-result { background: #f0f9ff; border-color: #4299e1; }
        .doc-result { background: #fffaf0; border-color: var(--warning); }
        
        .search-result h4, .news-result h4 {
            margin-bottom: 5px;
            font-size: 14px;
        }
        
        .search-result a, .news-result a {
            color: var(--secondary);
            text-decoration: none;
            font-size: 12px;
        }
        
        .search-result p, .news-result p {
            color: #666;
            font-size: 13px;
            margin-top: 5px;
        }
        
        .calc-result {
            background: linear-gradient(135deg, #f0fff4, #e6fffa);
            border-left: 4px solid var(--success);
            padding: 15px;
            border-radius: 8px;
        }
        
        .calc-result .expression { color: #666; font-size: 14px; }
        .calc-result .result { color: var(--success); font-size: 28px; font-weight: bold; }
        
        .stats-box {
            background: linear-gradient(135deg, #fefcbf, #faf089);
            border-radius: 12px;
            padding: 15px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .stat-item {
            background: white;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value { font-size: 22px; font-weight: bold; color: var(--primary); }
        .stat-label { font-size: 11px; color: #666; margin-top: 5px; }
        
        .error-message {
            background: #fff5f5;
            border-left: 4px solid var(--danger);
            color: #c53030;
            padding: 10px;
            border-radius: 8px;
        }
        
        .success-message {
            background: #f0fff4;
            border-left: 4px solid var(--success);
            color: #22543d;
            padding: 10px;
            border-radius: 8px;
        }
        
        .input-area {
            padding: 15px 25px;
            background: white;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 10px;
        }
        
        #userInput {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid var(--border);
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: all 0.3s;
        }
        
        #userInput:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        #sendBtn {
            padding: 12px 25px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        #sendBtn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .toast {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            animation: slideDown 0.3s ease-out;
            z-index: 1000;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .toast-success { background: var(--success); }
        .toast-error { background: var(--danger); }
        
        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @media (max-width: 768px) {
            .app-container {
                flex-direction: column;
                height: 100vh;
                border-radius: 0;
            }
            
            .sidebar {
                width: 100%;
                max-height: 200px;
            }
            
            .user-bubble { max-width: 85%; }
            .agent-bubble { max-width: 90%; }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sidebar for Document Management -->
        <div class="sidebar">
            <div class="sidebar-header">
                <h2>📚 Documents</h2>
                <p>Upload & manage your files</p>
            </div>
            
            <div class="upload-section">
                <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">
                        <strong>Click to upload</strong> or drag & drop<br>
                        <small>TXT, PDF, DOCX, CSV, MD, code files</small>
                    </div>
                </div>
                <input type="file" id="fileInput" multiple accept=".txt,.pdf,.docx,.csv,.md,.py,.js,.html,.css,.json,.xml" onchange="handleFileUpload(this.files)">
                
                <button class="btn btn-success" onclick="refreshDocuments()">
                    🔄 Refresh List
                </button>
                <button class="btn btn-danger" onclick="clearAllDocuments()">
                    🗑️ Clear All Documents
                </button>
            </div>
            
            <div class="document-list" id="documentList">
                <h3>📋 Your Documents</h3>
                <div class="empty-state" id="emptyState">
                    <div class="icon">📭</div>
                    <p>No documents yet</p>
                    <small>Upload files to get started</small>
                </div>
            </div>
        </div>
        
        <!-- Main Chat Area -->
        <div class="main-area">
            <div class="main-header">
                <h1>🤖 AI Agent Pro</h1>
                <small style="color: #666;">Web Search • Calculator • Document Query</small>
            </div>
            
            <div class="quick-actions">
                <button class="quick-btn" onclick="setQuickAction('search ')">🔍 Search</button>
                <button class="quick-btn" onclick="setQuickAction('news ')">📰 News</button>
                <button class="quick-btn" onclick="setQuickAction('calc ')">🧮 Calc</button>
                <button class="quick-btn" onclick="setQuickAction('convert ')">📏 Convert</button>
                <button class="quick-btn" onclick="setQuickAction('doc_stats')">📊 Stats</button>
                <button class="quick-btn" onclick="setQuickAction('list_docs')">📋 List</button>
                <button class="quick-btn" onclick="setQuickAction('help')">❓ Help</button>
                <button class="quick-btn" onclick="clearChat()">🗑️ Clear Chat</button>
            </div>
            
            <div class="chat-area" id="chatArea">
                <div class="message agent-message">
                    <div class="agent-bubble">
                        <strong>👋 Welcome to AI Agent Pro!</strong><br><br>
                        <strong>Quick Start:</strong><br>
                        • 🔍 <strong>Search web:</strong> "search Python tutorials"<br>
                        • 📰 <strong>Get news:</strong> "news technology"<br>
                        • 🧮 <strong>Calculate:</strong> "calc 15 * 3 + 27"<br>
                        • 📏 <strong>Convert:</strong> "convert 100 km to miles"<br>
                        • 📚 <strong>Documents:</strong> Upload files using the left panel<br><br>
                        <em>Type <strong>help</strong> for all commands</em>
                    </div>
                </div>
            </div>
            
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type your command or question..." 
                       onkeypress="if(event.key==='Enter') sendMessage()">
                <button id="sendBtn" onclick="sendMessage()">Send 🚀</button>
            </div>
        </div>
    </div>
    
    <script>
        // File upload handling
        const uploadArea = document.getElementById('uploadArea');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });
        
        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            handleFileUpload(files);
        }, false);
        
        async function handleFileUpload(files) {
            if (!files || files.length === 0) return;
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('file', file);
            }
            
            // Process multiple files
            for (let file of files) {
                const fileFormData = new FormData();
                fileFormData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: fileFormData
                    });
                    
                    const data = await response.json();
                    
                    if (data.type === 'document') {
                        showToast(data.message, 'success');
                        addMessage('agent', formatResponse(data));
                    } else {
                        showToast(data.message || 'Upload failed', 'error');
                    }
                } catch (error) {
                    showToast('Error uploading file: ' + file.name, 'error');
                }
            }
            
            refreshDocuments();
            document.getElementById('fileInput').value = '';
        }
        
        async function refreshDocuments() {
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query: 'list_docs' })
                });
                
                const data = await response.json();
                updateDocumentList(data);
            } catch (error) {
                console.error('Error refreshing documents:', error);
            }
        }
        
        function updateDocumentList(data) {
            const docList = document.getElementById('documentList');
            const emptyState = document.getElementById('emptyState');
            
            if (!data.files || data.files.length === 0) {
                docList.innerHTML = '<h3>📋 Your Documents</h3>' + emptyState.outerHTML;
                return;
            }
            
            let html = '<h3>📋 Your Documents</h3>';
            data.files.forEach(file => {
                const size = file.size ? (file.size / 1024).toFixed(1) + ' KB' : 'N/A';
                html += `
                    <div class="doc-item">
                        <div class="doc-item-header">
                            <span class="doc-name">📄 ${file.name}</span>
                            <span class="doc-type">${file.type || 'file'}</span>
                        </div>
                        <div class="doc-meta">
                            <span>📑 ${file.chunks} chunks</span>
                            <span>💾 ${size}</span>
                            <button class="remove-doc-btn" onclick="removeDocument('${file.name}')" title="Remove">✕</button>
                        </div>
                    </div>
                `;
            });
            
            docList.innerHTML = html;
        }
        
        async function removeDocument(filename) {
            if (!confirm(`Remove "${filename}" from document store?`)) return;
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query: `remove_doc ${filename}` })
                });
                
                const data = await response.json();
                showToast(data.message, data.type === 'error' ? 'error' : 'success');
                addMessage('agent', formatResponse(data));
                refreshDocuments();
            } catch (error) {
                showToast('Error removing document', 'error');
            }
        }
        
        async function clearAllDocuments() {
            if (!confirm('⚠️ Are you sure? This will remove ALL documents from the store and uploads folder.')) return;
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query: 'clear_all' })
                });
                
                const data = await response.json();
                showToast(data.message, 'success');
                addMessage('agent', formatResponse(data));
                refreshDocuments();
            } catch (error) {
                showToast('Error clearing documents', 'error');
            }
        }
        
        function showToast(message, type) {
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.textContent = message;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transition = 'opacity 0.3s';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }
        
        function setQuickAction(action) {
            document.getElementById('userInput').value = action;
            document.getElementById('userInput').focus();
        }
        
        function clearChat() {
            document.getElementById('chatArea').innerHTML = '';
        }
        
        function addMessage(type, content) {
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const bubble = document.createElement('div');
            bubble.className = `${type}-bubble`;
            bubble.innerHTML = content;
            
            messageDiv.appendChild(bubble);
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function showLoading() {
            const chatArea = document.getElementById('chatArea');
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message agent-message';
            loadingDiv.id = 'loadingMessage';
            loadingDiv.innerHTML = '<div class="agent-bubble"><div class="loading"></div> Processing...</div>';
            chatArea.appendChild(loadingDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
        }
        
        function hideLoading() {
            const loadingMsg = document.getElementById('loadingMessage');
            if (loadingMsg) loadingMsg.remove();
        }
        
        function formatResponse(data) {
            let html = '';
            
            switch(data.type) {
                case 'search':
                    html += `<strong>🔍 Results: "${data.query}"</strong> (${data.count} found)<br><br>`;
                    data.results.forEach((r, i) => {
                        html += `
                            <div class="search-result">
                                <h4>${i+1}. ${r.title}</h4>
                                <a href="${r.link}" target="_blank">🔗 ${r.link}</a>
                                <p>${r.snippet}</p>
                            </div>`;
                    });
                    break;
                
                case 'news':
                    html += `<strong>📰 News: "${data.query}"</strong> (${data.count} articles)<br><br>`;
                    data.results.forEach((r, i) => {
                        html += `
                            <div class="news-result">
                                <h4>${i+1}. ${r.title}</h4>
                                <small>📅 ${r.date} | 📰 ${r.source}</small><br>
                                <a href="${r.link}" target="_blank">🔗 Read more</a>
                                <p>${r.snippet}</p>
                            </div>`;
                    });
                    break;
                
                case 'calc':
                    html += `
                        <div class="calc-result">
                            <div class="expression">📝 ${data.expression}</div>
                            <div class="result">= ${data.result}</div>
                        </div>`;
                    break;
                
                case 'convert':
                    html += `<div class="calc-result">📏 <strong>${data.result}</strong></div>`;
                    break;
                
                case 'document':
                    html += `<div class="success-message">${data.message}</div>`;
                    break;
                
                case 'document_list':
                    html += `<strong>📋 Documents (${data.count}):</strong><br>`;
                    if (data.files && data.files.length > 0) {
                        data.files.forEach(f => {
                            html += `<div style="margin:5px 0;">📄 ${f.name} (${f.chunks} chunks, ${f.type})</div>`;
                        });
                    }
                    break;
                
                case 'document_search':
                    html += `<strong>📚 Found "${data.query}"</strong> (${data.count} matches)<br><br>`;
                    data.results.forEach(r => {
                        html += `
                            <div class="doc-result">
                                <strong>📄 ${r.file_name}</strong>
                                <span style="color:#ed8936;">(${(r.similarity*100).toFixed(1)}% match)</span>
                                <p>${r.text.substring(0, 200)}...</p>
                            </div>`;
                    });
                    break;
                
                case 'document_stats':
                    if (data.stats) {
                        const s = data.stats;
                        html += `
                            <div class="stats-box">
                                <strong>📊 Document Statistics</strong>
                                <div class="stats-grid">
                                    <div class="stat-item">
                                        <div class="stat-value">${s.total_documents}</div>
                                        <div class="stat-label">Documents</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">${s.total_chunks}</div>
                                        <div class="stat-label">Chunks</div>
                                    </div>
                                    <div class="stat-item">
                                        <div class="stat-value">${(s.total_size_chars/1000).toFixed(1)}K</div>
                                        <div class="stat-label">Characters</div>
                                    </div>
                                </div>
                                <br>
                                <strong>Types:</strong> ${s.file_types?.join(', ') || 'None'}<br>
                                <strong>Files:</strong><br>
                                ${s.files?.map(f => `📄 ${f.name} (${f.chunks} chunks)`).join('<br>') || 'No files'}
                            </div>`;
                    }
                    break;
                
                case 'help':
                    html += '<strong>📚 Commands:</strong><br><br>';
                    data.commands.forEach(c => {
                        html += `<div style="margin:8px 0;"><strong style="color:#667eea;">${c.cmd}</strong><br><small>${c.desc}</small><br><code>Example: ${c.example}</code></div>`;
                    });
                    break;
                
                case 'error':
                    html += `<div class="error-message">❌ ${data.message}</div>`;
                    break;
                
                default:
                    html += data.message || 'Done!';
            }
            
            return html;
        }
        
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            input.value = '';
            showLoading();
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ query: message })
                });
                
                const data = await response.json();
                hideLoading();
                addMessage('agent', formatResponse(data));
                
                // Refresh document list if document-related command
                if (['document', 'document_list', 'document_stats'].includes(data.type)) {
                    refreshDocuments();
                }
            } catch (error) {
                hideLoading();
                addMessage('agent', '<div class="error-message">❌ Connection error. Please try again.</div>');
            }
        }
        
        // Initial load
        refreshDocuments();
        document.getElementById('userInput').focus();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'type': 'error', 'message': 'No query provided'})
        
        response = agent.process_command(user_query)
        return jsonify(response)
    except Exception as e:
        return jsonify({'type': 'error', 'message': f'Server error: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'type': 'error', 'message': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'type': 'error', 'message': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'type': 'error', 'message': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'})
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        result = agent.document_store.add_document(file_path)
        
        if result['success']:
            return jsonify({
                'type': 'document',
                'message': result['message'],
                'filename': result['filename'],
                'chunks': result['chunks']
            })
        else:
            return jsonify({'type': 'error', 'message': result['message']})
    
    except Exception as e:
        return jsonify({'type': 'error', 'message': f'Upload error: {str(e)}'})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Production AI Agent Starting...")
    print("="*60)
    print("\n📱 Open: http://localhost:5000")
    print("\n✨ Features:")
    print("   • 📁 Drag & drop file upload")
    print("   • 📚 Document management panel")
    print("   • 🗑️ Clear all documents option")
    print("   • 🔍 Web search & news")
    print("   • 🧮 Calculator & converter")
    print("\n📋 Commands: search, news, calc, convert, add_doc, query_doc, doc_stats, list_docs, clear_all, help")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)