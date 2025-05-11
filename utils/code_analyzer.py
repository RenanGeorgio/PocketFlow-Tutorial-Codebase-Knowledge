import ast
import os
from typing import Dict, Set, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
from .call_llm import semantic_analizes

@dataclass
class RelationshipScore:
    direct_import: float = 0.0  # Direct import relationships
    inheritance: float = 0.0    # Class inheritance relationships  
    function_calls: float = 0.0 # Function/method calls between files
    semantic: float = 0.0       # Semantic similarity score
    
    def total(self) -> float:
        """Calculate weighted total score."""
        weights = {
            'direct_import': 0.3,
            'inheritance': 0.3,
            'function_calls': 0.2,
            'semantic': 0.2
        }
        return sum(getattr(self, k) * v for k, v in weights.items())

class CodeAnalyzer:
    def __init__(self):
        self._relationship_cache: Dict[str, Set[str]] = {}
        self._inheritance_cache: Dict[str, Set[str]] = {}
        self._score_cache: Dict[str, Dict[str, RelationshipScore]] = {}
        self._files_data: List[Tuple[str, str]] = []  # Store files data for semantic analysis

    @lru_cache(maxsize=1000)
    def analyze_file_relationships(self, files_data: List[Tuple[str, str]],
                                 file_patterns: Set[str], 
                                 target_list: Set[str] = None) -> Dict[str, float]:
        """Analyze relationships between files and target list.
        
        Args:
            files_data: List of (path, content) tuples
            file_patterns: Set of file patterns to consider
            target_list: Optional set of target patterns to prioritize
        """
        if not target_list:
            return {}
            
        self._files_data = files_data  # Store for semantic analysis
        priorities = {}
        
        # First pass - static analysis
        for path, content in files_data:
            score = self._analyze_file_relationships(path, content, target_list)
            
            if score.total() > 0:
                priorities[path] = score.total()
                
        return priorities
    
    def _analyze_file_relationships(self, path: str, content: str, target_list: Set[str]) -> RelationshipScore:
        """Analyze all types of relationships for a file."""
        score = RelationshipScore()
        
        try:
            # Language-specific analysis
            if path.endswith('.py'):
                tree = ast.parse(content)
                imports = self._extract_imports(tree)
                score.direct_import = self._calculate_import_score(imports, target_list)
                score.inheritance = self._analyze_inheritance_py(tree, target_list)
                score.function_calls = self._analyze_function_calls_py(tree, target_list)
            elif path.endswith(('.js', '.jsx', '.ts', '.tsx')):
                imports = self._extract_js_imports(content)
                score.direct_import = self._calculate_import_score(imports, target_list)
                score.inheritance = self._analyze_inheritance_js(content, target_list)
                score.function_calls = self._analyze_function_calls_js(content, target_list)
            elif path.endswith('.go'):
                imports = self._extract_go_imports(content)
                score.direct_import = self._calculate_import_score(imports, target_list)
                score.inheritance = self._analyze_inheritance_go(content, target_list)
                score.function_calls = self._analyze_function_calls_go(content, target_list)
            elif path.endswith('.java'):
                imports = self._extract_java_imports(content)
                score.direct_import = self._calculate_import_score(imports, target_list)
                score.inheritance = self._analyze_inheritance_java(content, target_list)
                score.function_calls = self._analyze_function_calls_java(content, target_list)
                
            # Only do semantic analysis if other scores are low
            if score.total() < 0.3:
                score.semantic = self._analyze_semantic_relationship(content, self._files_data, target_list)
                
        except Exception as e:
            print(f"Warning: Error analyzing {path}: {e}")
            
        return score

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from a Python AST."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        return imports
    
    def _extract_js_imports(self, content: str) -> Set[str]:
        """Extract imports from JavaScript/TypeScript code."""
        imports = set()
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ') or line.startswith('export '):
                if ' from ' in line:
                    module = line.split(' from ')[1].strip("'\"`;")
                    imports.add(module)
            elif 'require(' in line:
                start = line.find('require(') + 8
                end = line.find(')', start)
                if start > 7 and end > start:
                    module = line[start:end].strip("'\"")
                    imports.add(module)
        return imports
        
    def _extract_go_imports(self, content: str) -> Set[str]:
        """Extract imports from Go code."""
        imports = set()
        in_import_block = False
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import ('):
                in_import_block = True
            elif line == ')':
                in_import_block = False
            elif line.startswith('import ') or in_import_block:
                parts = line.split()
                if len(parts) > 1:
                    package = parts[-1].strip('"')
                    imports.add(package)
        return imports
        
    def _extract_java_imports(self, content: str) -> Set[str]:
        """Extract imports from Java code."""
        imports = set()
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                parts = line.replace('import', '').replace('static', '').strip().rstrip(';').split('.')
                if parts:
                    imports.add(parts[0])
        return imports

    def _calculate_import_score(self, imports: Set[str], target_list: Set[str]) -> float:
        """Calculate import relationship score."""
        direct_matches = sum(any(target.lower() in imp.lower() 
                           for target in target_list)
                           for imp in imports)
        return min(0.8, direct_matches * 0.2)  # Cap at 0.8
        
    def _analyze_inheritance_py(self, tree: ast.AST, target_list: Set[str]) -> float:
        """Analyze inheritance relationships in Python code."""
        score = 0.0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        if self._is_target_class(base.id, target_list):
                            score += 0.3
                    elif isinstance(base, ast.Attribute):
                        if self._is_target_class(base.attr, target_list):
                            score += 0.2
        return min(0.9, score)

    def _analyze_inheritance_js(self, content: str, target_list: Set[str]) -> float:
        """Analyze inheritance relationships in JavaScript/TypeScript code."""
        score = 0.0
        for line in content.split('\n'):
            line = line.strip()
            # Class inheritance
            if 'class' in line and 'extends' in line:
                parts = line.split('extends')
                if len(parts) > 1:
                    parent = parts[1].split('{')[0].strip()
                    if self._is_target_class(parent, target_list):
                        score += 0.3
            # React components
            elif '.extends(React.Component)' in line or 'extends Component' in line:
                score += 0.2
        return min(0.9, score)

    def _analyze_inheritance_go(self, content: str, target_list: Set[str]) -> float:
        """Analyze inheritance-like relationships in Go code."""
        score = 0.0
        # Look for struct embedding and interface implementation
        for line in content.split('\n'):
            line = line.strip()
            if 'type' in line and 'struct' in line:
                # Check for embedded types
                struct_content = line[line.find('struct'):].strip('{}')
                for embedded in struct_content.split('\n'):
                    embedded = embedded.strip()
                    if embedded and not ':' in embedded:  # Embedded type
                        if self._is_target_class(embedded, target_list):
                            score += 0.3
            elif 'type' in line and 'interface' in line:
                if any(self._is_target_class(t, target_list) for t in line.split()):
                    score += 0.2
        return min(0.9, score)

    def _analyze_inheritance_java(self, content: str, target_list: Set[str]) -> float:
        """Analyze inheritance relationships in Java code."""
        score = 0.0
        for line in content.split('\n'):
            line = line.strip()
            # Class inheritance
            if 'class' in line and 'extends' in line:
                parts = line.split('extends')
                if len(parts) > 1:
                    parent = parts[1].split('{')[0].strip().split(' ')[0]
                    if self._is_target_class(parent, target_list):
                        score += 0.3
            # Interface implementation
            elif 'implements' in line:
                parts = line.split('implements')
                if len(parts) > 1:
                    interfaces = parts[1].split('{')[0].strip().split(',')
                    if any(self._is_target_class(i.strip(), target_list) for i in interfaces):
                        score += 0.2
        return min(0.9, score)

    def _analyze_function_calls_py(self, tree: ast.AST, target_list: Set[str]) -> float:
        """Analyze function call relationships in Python code."""
        score = 0.0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if self._is_target_function(node.func.id, target_list):
                        score += 0.1
                elif isinstance(node.func, ast.Attribute):
                    if self._is_target_function(node.func.attr, target_list):
                        score += 0.1
        return min(0.7, score)

    def _analyze_function_calls_js(self, content: str, target_list: Set[str]) -> float:
        """Analyze function call relationships in JavaScript/TypeScript code."""
        score = 0.0
        for line in content.split('\n'):
            line = line.strip()
            # Look for function calls
            for target in target_list:
                target_lower = target.lower()
                if '.' in line and '(' in line:
                    func_name = line.split('(')[0].strip().split('.')[-1]
                    if target_lower in func_name.lower():
                        score += 0.1
                elif '(' in line:
                    func_name = line.split('(')[0].strip()
                    if target_lower in func_name.lower():
                        score += 0.1
        return min(0.7, score)

    def _analyze_function_calls_go(self, content: str, target_list: Set[str]) -> float:
        """Analyze function call relationships in Go code."""
        score = 0.0
        for line in content.split('\n'):
            line = line.strip()
            if '.' in line and '(' in line:
                # Method calls
                parts = line.split('(')[0].split('.')
                if len(parts) > 1 and self._is_target_function(parts[-1], target_list):
                    score += 0.1
            elif '(' in line:
                # Function calls
                func_name = line.split('(')[0].strip()
                if self._is_target_function(func_name, target_list):
                    score += 0.1
        return min(0.7, score)

    def _analyze_function_calls_java(self, content: str, target_list: Set[str]) -> float:
        """Analyze function call relationships in Java code."""
        score = 0.0
        for line in content.split('\n'):
            line = line.strip()
            if '.' in line and '(' in line:
                # Method calls
                method = line.split('(')[0].strip()
                if '.' in method:
                    method_name = method.split('.')[-1]
                    if self._is_target_function(method_name, target_list):
                        score += 0.1
        return min(0.7, score)

    def _analyze_semantic_relationship(self, content: str, files_data: List[Tuple[str, str]], target_list: Set[str]) -> float:
        """Analyze semantic relationships using LLM.
        
        Args:
            content: Content of the file being analyzed
            files_data: List of (path, content) tuples for all files
            target_list: Set of target patterns to prioritize
        """
        if not target_list:
            return 0.0

        # Create context of target files (but limit size for LLM context)
        target_context = ""
        for path, target_content in files_data:
            if any(target.lower() in path.lower() for target in target_list):
                # Add a truncated version of each target file
                preview = target_content[:1000] + "..." if len(target_content) > 1000 else target_content
                target_context += f"\nFile: {path}\n{preview}\n"

        if not target_context:
            return 0.0

        prompt = f"""Analyze the semantic relationship between the following code and the target files.
Focus on:
1. Shared domain concepts and terminology
2. Similar functionality or purpose
3. Architectural relationships
4. Business logic connections

Target files context:
{target_context}

Code to analyze:
{content[:1000]}... # Truncate for LLM context window

Rate the semantic relationship strength from 0.0 to 0.4, where:
0.0 = No semantic relationship
0.1-0.2 = Weak semantic relationship (few shared concepts)
0.3-0.4 = Strong semantic relationship (many shared concepts)

Output only the number (e.g. "0.2")."""

        try:
            response = semantic_analizes(prompt)
            score = float(response.strip())
            return min(max(score, 0.0), 0.4)  # Clamp between 0.0 and 0.4
        except (ValueError, TypeError):
            return 0.0  # Default to no relationship on error

    def _is_target_class(self, class_name: str, target_list: Set[str]) -> bool:
        """Check if a class name belongs to a target module."""
        return any(target.lower() in class_name.lower() for target in target_list)
    
    def _is_target_function(self, func_name: str, target_list: Set[str]) -> bool:
        """Check if a function belongs to a target module."""
        return any(target.lower() in func_name.lower() for target in target_list)
    
    def _find_related_file(self, import_name: str, current_dir: str, all_files: List[str], file_patterns: Set[str] = None) -> Optional[str]:
        """Find the source file for an imported name across all supported file types.
        
        Args:
            import_name: The name to search for (e.g., module name)
            current_dir: Current directory context
            all_files: List of all files in the workspace
            file_patterns: Set of file patterns to consider (e.g., "*.py", "*.js")
        """
        if file_patterns is None:
            # Default to common source file extensions
            file_patterns = {
                "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.java",
                "*.c", "*.cc", "*.cpp", "*.h"
            }
            
        # Extract extensions from patterns
        extensions = set()
        for pattern in file_patterns:
            if pattern.startswith("*."):
                extensions.add(pattern[1:])  # Remove the '*'
                
        # Generate potential paths for each extension
        potential_paths = []
        for ext in extensions:
            # Direct match with extension
            potential_paths.append(f"{import_name}{ext}")
            # Index file in directory
            potential_paths.extend([
                os.path.join(import_name, f"index{ext}"),
                os.path.join(import_name, f"main{ext}"),
                os.path.join(import_name, f"{os.path.basename(import_name)}{ext}")
            ])
            # Full path with extension
            potential_paths.append(os.path.join(current_dir, f"{import_name}{ext}"))
            
        # Special case for Python packages
        if "*.py" in file_patterns:
            potential_paths.extend([
                os.path.join(import_name, "__init__.py"),
                os.path.join(current_dir, import_name, "__init__.py")
            ])
            
        for file in all_files:
            if any(file.endswith(path) for path in potential_paths):
                return file
                
        return None
