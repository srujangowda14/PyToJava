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
 
# ── Python keywords ──────────────────────────────────────────────────────────
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
 
# ── Java keywords ─────────────────────────────────────────────────────────────
JAVA_KEYWORDS = {
    "class", "interface", "extends", "implements", "public", "private",
    "protected", "static", "final", "void", "return", "if", "else",
    "for", "while", "do", "try", "catch", "finally", "throw", "throws",
    "new", "this", "super", "import", "package", "null", "true", "false",
    "int", "long", "double", "float", "boolean", "char", "byte", "short",
    "String", "List", "Map", "Set", "ArrayList", "HashMap", "HashSet",
    "System", "out", "println", "Override", "abstract", "synchronized"
}