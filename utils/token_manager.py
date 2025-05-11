from typing import Dict, Set, List, Tuple, Any
import os
import tiktoken
from .code_analyzer import CodeAnalyzer

class TokenManager:
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 128000):
        """Initialize token manager.
        
        Args:
            model_name: Name of model to use for token counting
            max_tokens: Maximum tokens allowed in context
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.current_tokens = 0
        self.content_tokens: Dict[str, int] = {}  # Map content keys to token counts
        self.target_priorities: Dict[str, float] = {}  # Map paths to priority scores
        self.code_analyzer = CodeAnalyzer()  # Initialize code analyzer
        self.file_patterns = set()  # Store supported file patterns

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))

    def add_content(self, key: str, content: str) -> bool:
        """Add content to the token manager.
        
        Returns:
            bool: True if content was added, False if it would exceed token limit
        """
        tokens = self.count_tokens(content)
        if self.current_tokens + tokens > self.max_tokens:
            return False
        
        self.content_tokens[key] = tokens
        self.current_tokens += tokens
        return True

    def remove_content(self, key: str) -> None:
        """Remove content from the token manager."""
        if key in self.content_tokens:
            self.current_tokens -= self.content_tokens[key]
            del self.content_tokens[key]

    def get_available_tokens(self) -> int:
        """Get the number of tokens still available."""
        return self.max_tokens - self.current_tokens

    def set_file_patterns(self, patterns: Set[str]) -> None:
        """Set the supported file patterns for relationship analysis.
        
        Args:
            patterns: Set of glob patterns (e.g., {"*.py", "*.js"})
        """
        self.file_patterns = patterns

    def set_target_priorities(self, target_list: set) -> None:
        """Set priority scores for files based on target list and code relationships.
        
        Uses CodeAnalyzer to perform smarter priority scoring based on:
        - Direct target matches (1.0)
        - Import relationships (0.5-0.9)
        - Inheritance relationships (0.5-0.9)
        - Function call relationships (0.5-0.9)
        - Semantic relationships (0.1-0.4)
        
        Args:
            target_list: Set of target patterns to prioritize
        """
        self.target_priorities.clear()
        
        # Convert dictionary items to list of tuples for analyzer
        files_data = [(path, content) for path, content in self.content_tokens.items()]
        
        priorities = None
        if target_list:
            # Get relationship-based priorities with file patterns
            priorities = self.code_analyzer.analyze_file_relationships(
                files_data,
                self.file_patterns,
                target_list
            )
        
        # Update our priorities dictionary
        self.target_priorities.update(priorities)

    def create_hierarchical_context(self, files_data: List[Tuple[str, str]], 
                                  max_files_per_level: int = 50,
                                  target_list: set = None) -> Dict[str, Any]:
        """Create a hierarchical context from files data.
        
        This is the main entry point that handles both targeted and non-targeted analysis.
        When target_list is provided, it uses relationship-based prioritization.
        When target_list is None, it uses basic file-based prioritization.
        
        Args:
            files_data: List of (path, content) tuples
            max_files_per_level: Maximum number of files to include at each level
            target_list: Optional set of target patterns to prioritize
        """
        if target_list:
            return self._create_targeted_context(files_data, max_files_per_level, target_list)
        else:
            return self._create_basic_context(files_data, max_files_per_level)

    def _create_basic_context(self, files_data: List[Tuple[str, str]], 
                            max_files_per_level: int = 50) -> Dict[str, Any]:
        """Create basic hierarchical context without target analysis."""
        # Group files by directory level
        hierarchy: Dict[str, List[Tuple[str, str]]] = {}
        
        for path, content in files_data:
            depth = len(os.path.normpath(path).split(os.sep))
            if depth not in hierarchy:
                hierarchy[depth] = []
            hierarchy[depth].append((path, content))

        context = {
            "levels": {},
            "file_summaries": {},
            "total_files": len(files_data)
        }

        for depth in sorted(hierarchy.keys()):
            level_files = hierarchy[depth]
            
            # Basic sorting strategy
            level_files.sort(key=lambda x: (
                "test" in x[0].lower(),  # Deprioritize test files
                -len(x[1])  # Prioritize larger files
            ))

            # Take top N files for this level
            selected_files = level_files[:max_files_per_level]
            
            level_context = []
            for path, content in selected_files:
                if self.add_content(f"full_{path}", content):
                    level_context.append({
                        "path": path,
                        "type": "full",
                        "content": content
                    })
                else:
                    # If full content doesn't fit, add a summary
                    summary = self._create_file_summary(path, content)
                    if self.add_content(f"summary_{path}", summary):
                        level_context.append({
                            "path": path,
                            "type": "summary",
                            "content": summary
                        })
            
            if level_context:
                context["levels"][depth] = level_context

        return context

    def _create_targeted_context(self, files_data: List[Tuple[str, str]], 
                               max_files_per_level: int = 50,
                               target_list: set = None) -> Dict[str, Any]:
        """Create hierarchical context with target-based prioritization."""
        # Pre-analyze all files and set priorities
        for path, content in files_data:
            self.add_content(path, content)
        
        # Analyze relationships if we have file patterns
        if self.file_patterns:
            priorities = self.code_analyzer.analyze_file_relationships(
                files_data,
                target_list,
                self.file_patterns
            )
            self.target_priorities.update(priorities or {})

        # Group files by directory level
        hierarchy: Dict[str, List[Tuple[str, str]]] = {}
        for path, content in files_data:
            depth = len(os.path.normpath(path).split(os.sep))
            if depth not in hierarchy:
                hierarchy[depth] = []
            hierarchy[depth].append((path, content))

        context = {
            "levels": {},
            "file_summaries": {},
            "total_files": len(files_data),
            "target_focused_files": []  # Track files relevant to targets
        }

        for depth in sorted(hierarchy.keys()):
            level_files = hierarchy[depth]
            
            # Enhanced sorting with relationship priorities
            level_files.sort(key=lambda x: (
                -len([t for t in target_list if t.lower() in x[0].lower()]),  # Direct matches first
                -self.target_priorities.get(x[0], 0),  # Then relationship priority
                "test" in x[0].lower(),  # Then deprioritize test files
                -len(x[1])  # Finally, prioritize larger files
            ))

            selected_files = level_files[:max_files_per_level]
            
            level_context = []
            for path, content in selected_files:
                if self.add_content(f"full_{path}", content):
                    entry = {
                        "path": path,
                        "type": "full",
                        "content": content,
                        "priority": self.target_priorities.get(path, 0)
                    }
                    level_context.append(entry)
                    if entry["priority"] > 0:
                        context["target_focused_files"].append(path)
                else:
                    summary = self._create_file_summary(path, content)
                    if self.add_content(f"summary_{path}", summary):
                        entry = {
                            "path": path,
                            "type": "summary",
                            "content": summary,
                            "priority": self.target_priorities.get(path, 0)
                        }
                        level_context.append(entry)
                        if entry["priority"] > 0:
                            context["target_focused_files"].append(path)
            
            if level_context:
                context["levels"][depth] = level_context

        return context

    def _create_file_summary(self, path: str, content: str) -> str:
        """Create a summary of a file's content."""
        # Basic summary: first few lines and size info
        lines = content.split('\n')[:10]  # First 10 lines
        summary = f"File: {path}\n"
        summary += f"Size: {len(content)} chars\n"
        summary += f"Preview:\n{''.join(lines)}\n..."
        return summary
