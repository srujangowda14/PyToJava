from typing import List, Dict
import re

PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"
SOS_TOKEN   = "<SOS>"   # start of sequence
EOS_TOKEN   = "<EOS>"   # end of sequence
SEP_TOKEN   = "<SEP>"   # method separator
INDENT_TOK  = "<INDENT>"
DEDENT_TOK  = "<DEDENT>"
NEWLINE_TOK = "<NL>"
 
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN,
                  SEP_TOKEN, INDENT_TOK, DEDENT_TOK, NEWLINE_TOK]
 
# ── Python keywords 
PYTHON_KEYWORDS = {
    "def", "class", "return", "if", "elif", "else", "for", "while",
    "import", "from", "as", "try", "except", "finally", "with", "pass",
    "break", "continue", "raise", "assert", "lambda", "yield", "in",
    "not", "and", "or", "is", "None", "True", "False", "self",
    "super", "__init__", "__str__", "__repr__", "__len__", "print",
    "range", "len", "list", "dict", "set", "tuple", "str", "int",
    "float", "bool", "isinstance", "type", "append", "extend", "keys",
    "values", "items", "get", "pop", "update"
}
 
# ── Java keywords 
JAVA_KEYWORDS = {
    "class", "interface", "extends", "implements", "public", "private",
    "protected", "static", "final", "void", "return", "if", "else",
    "for", "while", "do", "try", "catch", "finally", "throw", "throws",
    "new", "this", "super", "import", "package", "null", "true", "false",
    "int", "long", "double", "float", "boolean", "char", "byte", "short",
    "String", "List", "Map", "Set", "ArrayList", "HashMap", "HashSet",
    "System", "out", "println", "Override", "abstract", "synchronized"
}

class CodeTokenizer:
    """
    Tokenizes Python/Java source code into sub-word tokens
    while preserving structural information via special tokens.
    Uses a simple but effective regex-based approach suitable
    for training a seq2seq model from scratch.
    """
 
    def __init__(self, lang: str = "python"):
        assert lang in ("python", "java"), f"Unsupported language: {lang}"
        self.lang = lang
        self.keywords = PYTHON_KEYWORDS if lang == "python" else JAVA_KEYWORDS
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def tokenize(self, code: str) -> List[str]:
        """Convert raw source code string → list of tokens."""
        if self._looks_pretokenized(code):
            return self._tokenize_pretokenized(code)
        if self.lang == "python":
            return self._tokenize_python(code)
        return self._tokenize_java(code)
 
    def detokenize(self, tokens: List[str]) -> str:
        """Best-effort conversion of token list back to source string."""
        return self._detokenize(tokens, self.lang)
 
    # ── Python tokenizer ──────────────────────────────────────────────────────
 
    def _tokenize_python(self, code: str) -> List[str]:
        tokens = []
        lines  = code.split("\n")
 
        indent_stack = [0]
 
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                tokens.append(NEWLINE_TOK)
                continue
 
            # Track indentation via INDENT / DEDENT
            current_indent = len(line) - len(line.lstrip())
            if current_indent > indent_stack[-1]:
                indent_stack.append(current_indent)
                tokens.append(INDENT_TOK)
            while current_indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(DEDENT_TOK)
 
            line_tokens = self._lex(stripped.strip())
            tokens.extend(line_tokens)
            tokens.append(NEWLINE_TOK)
 
        return tokens

    def _tokenize_pretokenized(self, code: str) -> List[str]:
        tokens = []
        legacy_map = {
            "NEW_LINE": NEWLINE_TOK,
            "INDENT": INDENT_TOK,
            "DEDENT": DEDENT_TOK,
        }
        for token in code.split():
            tokens.append(legacy_map.get(token, token))
        return tokens
 
    # ── Java tokenizer ────────────────────────────────────────────────────────
 
    def _tokenize_java(self, code: str) -> List[str]:
        tokens = []
        lines  = code.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                tokens.append(NEWLINE_TOK)
                continue
            tokens.extend(self._lex(stripped))
            tokens.append(NEWLINE_TOK)
        return tokens
 
    # ── Core lexer ────────────────────────────────────────────────────────────
 
    def _lex(self, text: str) -> List[str]:
        """Regex-based lexer; emits keywords, identifiers, literals, punctuation."""
        pattern = r"""
            (?P<STRING>  \"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|\"[^\"]*\"|\'[^\']*\') |
            (?P<COMMENT> \#.*$|//.*$)                                                     |
            (?P<NUMBER>  \b\d+\.?\d*\b)                                                   |
            (?P<IDENT>   \b[A-Za-z_]\w*\b)                                                |
            (?P<OP>      [+\-*/=<>!&|%^~]+|[(){}[\].,;:?])                               |
            (?P<SKIP>    \s+)
        """
        tokens = []
        for m in re.finditer(pattern, text, re.VERBOSE | re.MULTILINE):
            kind  = m.lastgroup
            value = m.group()
            if kind == "SKIP":
                continue
            elif kind == "COMMENT":
                continue          # strip comments (simplification)
            elif kind == "STRING":
                tokens.append("<STR>")   # collapse string literals
            elif kind == "NUMBER":
                tokens.append("<NUM>")   # collapse numeric literals
            else:
                tokens.append(value)
        return tokens

    def _looks_pretokenized(self, code: str) -> bool:
        markers = ("NEW_LINE", "INDENT", "DEDENT", "<NL>", "<INDENT>", "<DEDENT>")
        return any(marker in code for marker in markers) and "\n" not in code
 
    # ── Detokenizer ───────────────────────────────────────────────────────────
 
    def _detokenize(self, tokens: List[str], lang: str) -> str:
        lines   = []
        current = []
        indent  = 0

        for tok in tokens:
            if tok == NEWLINE_TOK:
                prefix = "    " * indent
                lines.append(prefix + " ".join(current))
                current = []
            elif tok == INDENT_TOK:
                indent += 1
            elif tok == DEDENT_TOK:
                indent = max(0, indent - 1)
            elif tok in (SOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
                continue
            else:
                current.append(tok)
 
        if current:
            lines.append(" ".join(current))
 
        return "\n".join(lines)
    
class Vocabulary:
    """
    Builds and manages token ↔ index mappings for both source and target.
    """
 
    def __init__(self):
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self._add_special_tokens()
 
    def _add_special_tokens(self):
        for tok in SPECIAL_TOKENS:
            self._add(tok)
 
    def _add(self, token: str):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx]   = token
 
    def build(self, token_lists: List[List[str]], min_freq: int = 2):
        """Build vocab from a list of token sequences with min frequency threshold."""
        from collections import Counter
        freq: Counter = Counter()
        for seq in token_lists:
            freq.update(seq)
        for token, count in freq.items():
            if count >= min_freq:
                self._add(token)
        print(f"[Vocab] Built vocabulary of size {len(self)} "
              f"(min_freq={min_freq})")
 
    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(t, unk) for t in tokens]
 
    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token.get(i, UNK_TOKEN) for i in indices]
 
    def __len__(self):
        return len(self.token2idx)
 
    @property
    def pad_idx(self) -> int:
        return self.token2idx[PAD_TOKEN]
 
    @property
    def sos_idx(self) -> int:
        return self.token2idx[SOS_TOKEN]
 
    @property
    def eos_idx(self) -> int:
        return self.token2idx[EOS_TOKEN]
