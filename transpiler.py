#!/usr/bin/env python3
"""
Simple Language Transpiler v3 - AST Edition
Production-grade transpiler with Abstract Syntax Tree parsing and HPP support
"""

import re
import sys
import os
from typing import List, Dict, Optional, Tuple, Set, Union, Any
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# ============================================================================
# TOKEN SYSTEM
# ============================================================================

class TokenType(Enum):
    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING" 
    IDENTIFIER = "IDENTIFIER"
    
    # Keywords
    USE = "USE"
    FN = "FN"
    IF = "IF"
    ELIF = "ELIF"
    ELSE = "ELSE"
    WHILE = "WHILE"
    FOR = "FOR"
    IN = "IN"
    LOOP = "LOOP"
    END = "END"
    RETURN = "RETURN"
    SAY = "SAY"
    CLASS = "CLASS"
    NEW = "NEW"
    SELF = "SELF"
    
    # Operators
    ASSIGN = "ASSIGN"
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    LESS_THAN = "LESS_THAN"
    GREATER_THAN = "GREATER_THAN"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER_EQUAL = "GREATER_EQUAL"
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    
    # Delimiters
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    COMMA = "COMMA"
    DOT = "DOT"
    COLON = "COLON"
    
    # Special
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    INDENT = "INDENT"
    DEDENT = "DEDENT"

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class SimpleError(Exception):
    """Base exception for Simple language errors"""
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(f"Line {line}, Column {column}: {message}")

# ============================================================================
# LEXER (TOKENIZER)
# ============================================================================

class SimpleLexer:
    """Production-grade lexer for Simple language"""
    
    KEYWORDS = {
        'use': TokenType.USE,
        'fn': TokenType.FN,
        'if': TokenType.IF,
        'elif': TokenType.ELIF,
        'else': TokenType.ELSE,
        'while': TokenType.WHILE,
        'for': TokenType.FOR,
        'in': TokenType.IN,
        'loop': TokenType.LOOP,
        'end': TokenType.END,
        'return': TokenType.RETURN,
        'say': TokenType.SAY,
        'class': TokenType.CLASS,
        'new': TokenType.NEW,
        'self': TokenType.SELF,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
    }
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.indent_stack = [0]  # Track indentation levels
        
    def error(self, message: str):
        raise SimpleError(message, self.line, self.column)
        
    def peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos >= len(self.text):
            return '\0'
        return self.text[pos]
        
    def advance(self) -> str:
        if self.pos >= len(self.text):
            return '\0'
        
        char = self.text[self.pos]
        self.pos += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
            
        return char
        
    def skip_whitespace_inline(self):
        """Skip spaces and tabs, but not newlines"""
        while self.peek() in ' \t':
            self.advance()
            
    def read_string(self) -> str:
        quote_char = self.advance()  # Skip opening quote
        value = ''
        
        while self.peek() != quote_char and self.peek() != '\0':
            if self.peek() == '\\':
                self.advance()  # Skip backslash
                escaped = self.advance()
                if escaped == 'n':
                    value += '\n'
                elif escaped == 't':
                    value += '\t'
                elif escaped == '\\':
                    value += '\\'
                elif escaped == quote_char:
                    value += quote_char
                else:
                    value += escaped
            else:
                value += self.advance()
                
        if self.peek() == '\0':
            self.error("Unterminated string literal")
            
        self.advance()  # Skip closing quote
        return value
        
    def read_number(self) -> str:
        value = ''
        has_dot = False
        
        # Handle hex numbers
        if self.peek() == '0' and self.peek(1) in 'xX':
            value += self.advance()  # '0'
            value += self.advance()  # 'x' or 'X'
            while self.peek().lower() in '0123456789abcdef':
                value += self.advance()
            return value
        
        # Handle decimal numbers
        while self.peek().isdigit() or (self.peek() == '.' and not has_dot):
            if self.peek() == '.':
                has_dot = True
            value += self.advance()
            
        return value
        
    def read_identifier(self) -> str:
        value = ''
        
        while self.peek().isalnum() or self.peek() == '_':
            value += self.advance()
            
        return value
        
    def handle_indentation(self):
        """Handle indentation-based block structure"""
        indent_level = 0
        while self.peek() in ' \t':
            if self.peek() == ' ':
                indent_level += 1
            else:  # tab
                indent_level += 4  # Treat tab as 4 spaces
            self.advance()
            
        current_indent = self.indent_stack[-1]
        
        if indent_level > current_indent:
            # Increased indentation
            self.indent_stack.append(indent_level)
            self.tokens.append(Token(TokenType.INDENT, '', self.line, self.column))
        elif indent_level < current_indent:
            # Decreased indentation
            while self.indent_stack and self.indent_stack[-1] > indent_level:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, '', self.line, self.column))
            
            if not self.indent_stack or self.indent_stack[-1] != indent_level:
                self.error("Invalid indentation level")
        
        # Same indentation level - no tokens needed
        
    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            # Handle newlines and indentation
            if self.peek() == '\n':
                self.advance()
                self.tokens.append(Token(TokenType.NEWLINE, '\\n', self.line - 1, self.column))
                
                # Check for indentation on next line
                if self.pos < len(self.text) and self.peek() in ' \t':
                    self.handle_indentation()
                continue
                
            # Skip inline whitespace
            self.skip_whitespace_inline()
            
            if self.pos >= len(self.text):
                break
                
            char = self.peek()
            start_line = self.line
            start_column = self.column
            
            # Skip comments
            if char == '#':
                while self.peek() != '\n' and self.peek() != '\0':
                    self.advance()
                continue
                
            # String literals
            if char in '"\'':
                value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, f'"{value}"', start_line, start_column))
                continue
                
            # Numbers
            if char.isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, value, start_line, start_column))
                continue
                
            # Identifiers and keywords
            if char.isalpha() or char == '_':
                value = self.read_identifier()
                token_type = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
                self.tokens.append(Token(token_type, value, start_line, start_column))
                continue
                
            # Two-character operators
            two_char_ops = {
                '==': TokenType.EQUALS,
                '!=': TokenType.NOT_EQUALS,
                '<=': TokenType.LESS_EQUAL,
                '>=': TokenType.GREATER_EQUAL,
            }
            
            two_char = char + self.peek(1)
            if two_char in two_char_ops:
                self.advance()
                self.advance()
                self.tokens.append(Token(two_char_ops[two_char], two_char, start_line, start_column))
                continue
                
            # Single-character tokens
            single_char_tokens = {
                '=': TokenType.ASSIGN,
                '<': TokenType.LESS_THAN,
                '>': TokenType.GREATER_THAN,
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.MULTIPLY,
                '/': TokenType.DIVIDE,
                '%': TokenType.MODULO,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                '[': TokenType.LBRACKET,
                ']': TokenType.RBRACKET,
                '{': TokenType.LBRACE,
                '}': TokenType.RBRACE,
                ',': TokenType.COMMA,
                '.': TokenType.DOT,
                ':': TokenType.COLON,
            }
            
            if char in single_char_tokens:
                self.advance()
                self.tokens.append(Token(single_char_tokens[char], char, start_line, start_column))
                continue
                
            # Unknown character
            self.error(f"Unexpected character: '{char}'")
            
        # Add final DEDENT tokens
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, '', self.line, self.column))
            
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens

# ============================================================================
# ABSTRACT SYNTAX TREE NODES
# ============================================================================

class ASTNode(ABC):
    """Base class for all AST nodes"""
    pass

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

@dataclass
class UseStatement(ASTNode):
    library: str

@dataclass
class FunctionDef(ASTNode):
    name: str
    parameters: List[str]
    body: List[ASTNode]

@dataclass
class ClassDef(ASTNode):
    name: str
    body: List[ASTNode]

@dataclass
class Assignment(ASTNode):
    target: str
    value: ASTNode

@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    operator: TokenType
    right: ASTNode

@dataclass
class UnaryOp(ASTNode):
    operator: TokenType
    operand: ASTNode

@dataclass
class FunctionCall(ASTNode):
    name: str
    arguments: List[ASTNode]

@dataclass
class MethodCall(ASTNode):
    object: ASTNode
    method: str
    arguments: List[ASTNode]

@dataclass
class Identifier(ASTNode):
    name: str

@dataclass
class Literal(ASTNode):
    value: Union[int, float, str]
    type: str  # 'int', 'float', 'string'

@dataclass
class ArrayLiteral(ASTNode):
    elements: List[ASTNode]

@dataclass
class DictLiteral(ASTNode):
    pairs: List[Tuple[ASTNode, ASTNode]]

@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    then_body: List[ASTNode]
    elif_parts: List[Tuple[ASTNode, List[ASTNode]]]  # (condition, body) pairs
    else_body: Optional[List[ASTNode]]

@dataclass
class WhileLoop(ASTNode):
    condition: ASTNode
    body: List[ASTNode]

@dataclass
class ForLoop(ASTNode):
    variable: str
    iterable: ASTNode
    body: List[ASTNode]

@dataclass
class CountLoop(ASTNode):
    variable: str
    start: ASTNode
    end: ASTNode
    body: List[ASTNode]

@dataclass
class ReturnStatement(ASTNode):
    value: Optional[ASTNode]

@dataclass
class SayStatement(ASTNode):
    expressions: List[ASTNode]

@dataclass
class Block(ASTNode):
    statements: List[ASTNode]

# ============================================================================
# PARSER
# ============================================================================

class SimpleParser:
    """Production-grade recursive descent parser"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else Token(TokenType.EOF, '', 0, 0)
        
    def error(self, message: str):
        raise SimpleError(message, self.current_token.line, self.current_token.column)
        
    def advance(self):
        """Move to next token"""
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
            self.current_token = self.tokens[self.pos]
            
    def peek(self, offset: int = 1) -> Token:
        """Look ahead at token"""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return Token(TokenType.EOF, '', 0, 0)
        
    def match(self, *token_types: TokenType) -> bool:
        """Check if current token matches any of the given types"""
        return self.current_token.type in token_types
        
    def consume(self, token_type: TokenType, message: str = "") -> Token:
        """Consume token of expected type or raise error"""
        if self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        
        if not message:
            message = f"Expected {token_type.value}, got {self.current_token.type.value}"
        self.error(message)
        
    def skip_newlines(self):
        """Skip newline tokens"""
        while self.match(TokenType.NEWLINE):
            self.advance()
            
    def parse(self) -> Program:
        """Parse the entire program"""
        statements = []
        
        self.skip_newlines()
        
        while not self.match(TokenType.EOF):
            if self.match(TokenType.NEWLINE):
                self.advance()
                continue
                
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
                
        return Program(statements)
        
    def parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement"""
        self.skip_newlines()
        
        # Skip INDENT/DEDENT tokens at statement level
        if self.match(TokenType.INDENT, TokenType.DEDENT):
            self.advance()
            return self.parse_statement()
        
        if self.match(TokenType.USE):
            return self.parse_use_statement()
        elif self.match(TokenType.FN):
            return self.parse_function_def()
        elif self.match(TokenType.CLASS):
            return self.parse_class_def()
        elif self.match(TokenType.IF):
            return self.parse_if_statement()
        elif self.match(TokenType.WHILE):
            return self.parse_while_loop()
        elif self.match(TokenType.FOR):
            return self.parse_for_loop()
        elif self.match(TokenType.LOOP):
            return self.parse_count_loop()
        elif self.match(TokenType.RETURN):
            return self.parse_return_statement()
        elif self.match(TokenType.SAY):
            return self.parse_say_statement()
        elif self.match(TokenType.IDENTIFIER):
            # Could be assignment or expression statement
            if self.peek().type == TokenType.ASSIGN:
                return self.parse_assignment()
            else:
                # Expression statement (function call, etc.)
                expr = self.parse_expression()
                self.skip_newlines()
                return expr
        elif self.match(TokenType.EOF, TokenType.END):
            return None
        else:
            self.error(f"Unexpected token: {self.current_token.value}")
            
    def parse_use_statement(self) -> UseStatement:
        """Parse use statement"""
        self.consume(TokenType.USE)
        library_token = self.consume(TokenType.STRING)
        library = library_token.value[1:-1]  # Remove quotes
        self.skip_newlines()
        return UseStatement(library)
        
    def parse_function_def(self) -> FunctionDef:
        """Parse function definition"""
        self.consume(TokenType.FN)
        name_token = self.consume(TokenType.IDENTIFIER)
        name = name_token.value
        
        # Parse parameters
        parameters = []
        while self.match(TokenType.IDENTIFIER):
            param_token = self.consume(TokenType.IDENTIFIER)
            parameters.append(param_token.value)
            
        self.skip_newlines()
        
        # Parse body
        body = self.parse_block()
        
        return FunctionDef(name, parameters, body)
        
    def parse_class_def(self) -> ClassDef:
        """Parse class definition"""
        self.consume(TokenType.CLASS)
        name_token = self.consume(TokenType.IDENTIFIER)
        name = name_token.value
        
        self.skip_newlines()
        
        # Parse body
        body = self.parse_block()
        
        return ClassDef(name, body)
        
    def parse_block(self) -> List[ASTNode]:
        """Parse a block of statements"""
        statements = []
        
        self.skip_newlines()
        
        # Handle indented block
        if self.match(TokenType.INDENT):
            self.advance()
            
            while not self.match(TokenType.DEDENT, TokenType.EOF, TokenType.END):
                if self.match(TokenType.NEWLINE):
                    self.advance()
                    continue
                    
                stmt = self.parse_statement()
                if stmt:
                    statements.append(stmt)
            
            # Consume DEDENT or END
            if self.match(TokenType.DEDENT):
                self.advance()
            
        # Handle END-terminated block (no indentation)
        while not self.match(TokenType.END, TokenType.EOF, TokenType.ELIF, TokenType.ELSE):
            if self.match(TokenType.NEWLINE):
                self.advance()
                continue
                
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        
        # Consume 'end' if present
        if self.match(TokenType.END):
            self.advance()
            self.skip_newlines()
                
        return statements
        
    def parse_assignment(self) -> Assignment:
        """Parse assignment statement"""
        target_token = self.consume(TokenType.IDENTIFIER)
        target = target_token.value
        self.consume(TokenType.ASSIGN)
        value = self.parse_expression()
        self.skip_newlines()
        return Assignment(target, value)
        
    def parse_if_statement(self) -> IfStatement:
        """Parse if statement with elif and else"""
        self.consume(TokenType.IF)
        condition = self.parse_expression()
        self.skip_newlines()
        
        then_body = self.parse_block()
        
        elif_parts = []
        while self.match(TokenType.ELIF):
            self.advance()
            elif_condition = self.parse_expression()
            self.skip_newlines()
            elif_body = self.parse_block()
            elif_parts.append((elif_condition, elif_body))
            
        else_body = None
        if self.match(TokenType.ELSE):
            self.advance()
            self.skip_newlines()
            else_body = self.parse_block()
            
        return IfStatement(condition, then_body, elif_parts, else_body)
        
    def parse_while_loop(self) -> WhileLoop:
        """Parse while loop"""
        self.consume(TokenType.WHILE)
        condition = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()
        return WhileLoop(condition, body)
        
    def parse_for_loop(self) -> ForLoop:
        """Parse for-in loop"""
        self.consume(TokenType.FOR)
        var_token = self.consume(TokenType.IDENTIFIER)
        variable = var_token.value
        self.consume(TokenType.IN)
        iterable = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()
        return ForLoop(variable, iterable, body)
        
    def parse_count_loop(self) -> CountLoop:
        """Parse count loop"""
        self.consume(TokenType.LOOP)
        var_token = self.consume(TokenType.IDENTIFIER)
        variable = var_token.value
        start = self.parse_expression()
        end = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()
        return CountLoop(variable, start, end, body)
        
    def parse_return_statement(self) -> ReturnStatement:
        """Parse return statement"""
        self.consume(TokenType.RETURN)
        value = None
        if not self.match(TokenType.NEWLINE, TokenType.EOF):
            value = self.parse_expression()
        self.skip_newlines()
        return ReturnStatement(value)
        
    def parse_say_statement(self) -> SayStatement:
        """Parse say statement"""
        self.consume(TokenType.SAY)
        expressions = []
        
        if not self.match(TokenType.NEWLINE, TokenType.EOF):
            expressions.append(self.parse_expression())
            
            while self.match(TokenType.COMMA):
                self.advance()
                expressions.append(self.parse_expression())
                
        self.skip_newlines()
        return SayStatement(expressions)
        
    def parse_expression(self) -> ASTNode:
        """Parse expression with operator precedence"""
        return self.parse_logical_or()
        
    def parse_logical_or(self) -> ASTNode:
        """Parse logical OR expressions"""
        expr = self.parse_logical_and()
        
        while self.match(TokenType.OR):
            operator = self.current_token.type
            self.advance()
            right = self.parse_logical_and()
            expr = BinaryOp(expr, operator, right)
            
        return expr
        
    def parse_logical_and(self) -> ASTNode:
        """Parse logical AND expressions"""
        expr = self.parse_equality()
        
        while self.match(TokenType.AND):
            operator = self.current_token.type
            self.advance()
            right = self.parse_equality()
            expr = BinaryOp(expr, operator, right)
            
        return expr
        
    def parse_equality(self) -> ASTNode:
        """Parse equality expressions"""
        expr = self.parse_comparison()
        
        while self.match(TokenType.EQUALS, TokenType.NOT_EQUALS):
            operator = self.current_token.type
            self.advance()
            right = self.parse_comparison()
            expr = BinaryOp(expr, operator, right)
            
        return expr
        
    def parse_comparison(self) -> ASTNode:
        """Parse comparison expressions"""
        expr = self.parse_addition()
        
        while self.match(TokenType.LESS_THAN, TokenType.GREATER_THAN, 
                         TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL):
            operator = self.current_token.type
            self.advance()
            right = self.parse_addition()
            expr = BinaryOp(expr, operator, right)
            
        return expr
        
    def parse_addition(self) -> ASTNode:
        """Parse addition and subtraction"""
        expr = self.parse_multiplication()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.current_token.type
            self.advance()
            right = self.parse_multiplication()
            expr = BinaryOp(expr, operator, right)
            
        return expr
        
    def parse_multiplication(self) -> ASTNode:
        """Parse multiplication, division, and modulo"""
        expr = self.parse_unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self.current_token.type
            self.advance()
            right = self.parse_unary()
            expr = BinaryOp(expr, operator, right)
            
        return expr
        
    def parse_unary(self) -> ASTNode:
        """Parse unary expressions"""
        if self.match(TokenType.NOT, TokenType.MINUS):
            operator = self.current_token.type
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(operator, operand)
            
        return self.parse_postfix()
        
    def parse_postfix(self) -> ASTNode:
        """Parse postfix expressions (method calls, array access)"""
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.DOT):
                self.advance()
                method_token = self.consume(TokenType.IDENTIFIER)
                method = method_token.value
                
                if self.match(TokenType.LPAREN):
                    # Method call
                    self.advance()
                    arguments = []
                    
                    if not self.match(TokenType.RPAREN):
                        arguments.append(self.parse_expression())
                        while self.match(TokenType.COMMA):
                            self.advance()
                            arguments.append(self.parse_expression())
                            
                    self.consume(TokenType.RPAREN)
                    expr = MethodCall(expr, method, arguments)
                else:
                    # Property access (treat as method call with no args)
                    expr = MethodCall(expr, method, [])
                    
            elif self.match(TokenType.LBRACKET):
                # Array access
                self.advance()
                index = self.parse_expression()
                self.consume(TokenType.RBRACKET)
                expr = MethodCall(expr, "[]", [index])  # Treat as special method
                
            elif self.match(TokenType.LPAREN) and isinstance(expr, Identifier):
                # Function call
                self.advance()
                arguments = []
                
                if not self.match(TokenType.RPAREN):
                    arguments.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        self.advance()
                        arguments.append(self.parse_expression())
                        
                self.consume(TokenType.RPAREN)
                expr = FunctionCall(expr.name, arguments)
                
            else:
                break
                
        return expr
        
    def parse_primary(self) -> ASTNode:
        """Parse primary expressions"""
        if self.match(TokenType.NUMBER):
            value = self.current_token.value
            self.advance()
            
            if '.' in value or 'e' in value.lower():
                return Literal(float(value), 'float')
            elif value.startswith('0x'):
                return Literal(int(value, 16), 'int')
            else:
                return Literal(int(value), 'int')
                
        elif self.match(TokenType.STRING):
            value = self.current_token.value[1:-1]  # Remove quotes
            self.advance()
            return Literal(value, 'string')
            
        elif self.match(TokenType.IDENTIFIER):
            name = self.current_token.value
            self.advance()
            return Identifier(name)
            
        elif self.match(TokenType.NEW):
            self.advance()
            class_name = self.consume(TokenType.IDENTIFIER).value
            
            arguments = []
            if self.match(TokenType.LPAREN):
                self.advance()
                if not self.match(TokenType.RPAREN):
                    arguments.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        self.advance()
                        arguments.append(self.parse_expression())
                self.consume(TokenType.RPAREN)
                
            return FunctionCall(f"new {class_name}", arguments)
            
        elif self.match(TokenType.LBRACKET):
            # Array literal
            self.advance()
            elements = []
            
            if not self.match(TokenType.RBRACKET):
                elements.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    self.advance()
                    elements.append(self.parse_expression())
                    
            self.consume(TokenType.RBRACKET)
            return ArrayLiteral(elements)
            
        elif self.match(TokenType.LBRACE):
            # Dictionary literal
            self.advance()
            pairs = []
            
            if not self.match(TokenType.RBRACE):
                key = self.parse_expression()
                self.consume(TokenType.COLON)
                value = self.parse_expression()
                pairs.append((key, value))
                
                while self.match(TokenType.COMMA):
                    self.advance()
                    key = self.parse_expression()
                    self.consume(TokenType.COLON)
                    value = self.parse_expression()
                    pairs.append((key, value))
                    
            self.consume(TokenType.RBRACE)
            return DictLiteral(pairs)
            
        elif self.match(TokenType.LPAREN):
            # Parenthesized expression
            self.advance()
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN)
            return expr
            
        else:
            self.error(f"Unexpected token in expression: {self.current_token.value}")

# ============================================================================
# CODE GENERATOR
# ============================================================================

class CodeGenerator:
    """Generate C++ or HPP code from AST"""
    
    def __init__(self, output_type: str = "cpp"):
        self.output_type = output_type.lower()  # "cpp" or "hpp"
        self.includes = set()
        self.forward_declarations = []
        self.global_vars = []
        self.functions = []
        self.classes = []
        self.indent_level = 0
        
    def indent(self) -> str:
        return "    " * self.indent_level
        
    def generate(self, ast: Program) -> str:
        """Generate code from AST"""
        # First pass: collect declarations
        for stmt in ast.statements:
            self.collect_declarations(stmt)
            
        # Second pass: generate code
        for stmt in ast.statements:
            self.generate_statement(stmt)
            
        return self.build_output()
        
    def collect_declarations(self, node: ASTNode):
        """Collect function and class declarations"""
        if isinstance(node, FunctionDef):
            if node.name != 'main':
                params = ', '.join(f'auto {param}' for param in node.parameters)
                if params:
                    self.forward_declarations.append(f'auto {node.name}({params});')
                else:
                    self.forward_declarations.append(f'auto {node.name}();')
                    
        elif isinstance(node, ClassDef):
            # Classes are declared inline, no forward declaration needed
            pass
            
    def generate_statement(self, node: ASTNode):
        """Generate code for a statement"""
        if isinstance(node, UseStatement):
            self.includes.add(f'#include <{node.library}>')
            if node.library in ['iostream', 'string', 'vector', 'map', 'memory', 'any']:
                self.includes.add('using namespace std;')
                
        elif isinstance(node, Assignment):
            if self.indent_level == 0:  # Global variable
                value_code = self.generate_expression(node.value)
                if isinstance(node.value, ArrayLiteral):
                    if not node.value.elements:
                        self.global_vars.append(f'vector<double> {node.target};')
                    else:
                        self.global_vars.append(f'vector<double> {node.target} = {value_code};')
                elif isinstance(node.value, DictLiteral):
                    self.global_vars.append(f'map<string, any> {node.target} = {value_code};')
                else:
                    self.global_vars.append(f'auto {node.target} = {value_code};')
            else:
                # Local assignment - handled in function context
                pass
                
        elif isinstance(node, FunctionDef):
            self.generate_function(node)
            
        elif isinstance(node, ClassDef):
            self.generate_class(node)
            
    def generate_function(self, node: FunctionDef):
        """Generate function code"""
        lines = []
        
        if node.name == 'main':
            if self.output_type == "hpp":
                return  # Don't include main in header
            lines.append('int main() {')
        else:
            params = ', '.join(f'auto {param}' for param in node.parameters)
            if params:
                lines.append(f'auto {node.name}({params}) {{')
            else:
                lines.append(f'auto {node.name}() {{')
                
        self.indent_level += 1
        
        for stmt in node.body:
            stmt_code = self.generate_statement_code(stmt)
            if stmt_code:
                lines.append(f'{self.indent()}{stmt_code}')
                
        self.indent_level -= 1
        
        if node.name == 'main':
            lines.append('    return 0;')
            
        lines.append('}')
        
        if self.output_type == "hpp":
            # Inline functions in header
            self.functions.append('\n'.join(lines))
        else:
            self.functions.append('\n'.join(lines))
            
    def generate_class(self, node: ClassDef):
        """Generate class code"""
        lines = []
        lines.append(f'class {node.name} {{')
        lines.append('public:')
        
        # Separate member variables and methods
        member_vars = []
        methods = []
        
        for stmt in node.body:
            if isinstance(stmt, Assignment):
                value_code = self.generate_expression(stmt.value)
                member_vars.append(f'    auto {stmt.target} = {value_code};')
            elif isinstance(stmt, FunctionDef):
                method_code = self.generate_method(stmt, node.name)
                methods.append(method_code)
                
        # Add methods
        for method in methods:
            lines.extend(method.split('\n'))
            
        # Add member variables in private section
        if member_vars:
            lines.append('private:')
            lines.extend(member_vars)
            
        lines.append('};')
        self.classes.append('\n'.join(lines))
        
    def generate_method(self, node: FunctionDef, class_name: str) -> str:
        """Generate method code"""
        lines = []
        
        if node.name == 'constructor':
            # Constructor
            params = ', '.join(f'auto {param}' for param in node.parameters)
            if params:
                lines.append(f'    {class_name}({params}) {{')
            else:
                lines.append(f'    {class_name}() {{')
        else:
            # Regular method
            params = ', '.join(f'auto {param}' for param in node.parameters)
            if params:
                lines.append(f'    auto {node.name}({params}) {{')
            else:
                lines.append(f'    auto {node.name}() {{')
                
        self.indent_level += 2
        
        for stmt in node.body:
            stmt_code = self.generate_statement_code(stmt)
            if stmt_code:
                # Convert self to this->
                stmt_code = stmt_code.replace('self.', 'this->')
                lines.append(f'{self.indent()}{stmt_code}')
                
        self.indent_level -= 2
        lines.append('    }')
        
        return '\n'.join(lines)
        
    def generate_statement_code(self, node: ASTNode) -> str:
        """Generate code for statement within function"""
        if isinstance(node, Assignment):
            value_code = self.generate_expression(node.value)
            if isinstance(node.value, ArrayLiteral) and not node.value.elements:
                return f'vector<double> {node.target};'
            else:
                return f'auto {node.target} = {value_code};'
            
        elif isinstance(node, SayStatement):
            if not node.expressions:
                return 'cout << endl;'
            
            parts = []
            for expr in node.expressions:
                if isinstance(expr, Literal) and expr.type == 'string':
                    # Handle string interpolation
                    interpolated = self.handle_string_interpolation(expr.value)
                    parts.extend(interpolated)
                else:
                    expr_code = self.generate_expression(expr)
                    parts.append(expr_code)
                
            return f'cout << {" << ".join(parts)} << endl;'
            
        elif isinstance(node, ReturnStatement):
            if node.value:
                value_code = self.generate_expression(node.value)
                return f'return {value_code};'
            else:
                return 'return;'
                
        elif isinstance(node, IfStatement):
            return self.generate_if_statement(node)
            
        elif isinstance(node, WhileLoop):
            condition_code = self.generate_expression(node.condition)
            lines = [f'while ({condition_code}) {{']
            
            self.indent_level += 1
            for stmt in node.body:
                stmt_code = self.generate_statement_code(stmt)
                if stmt_code:
                    lines.append(f'{self.indent()}{stmt_code}')
            self.indent_level -= 1
            
            lines.append(f'{self.indent()}}}')
            return '\n'.join(lines)
            
        elif isinstance(node, ForLoop):
            var = node.variable
            iterable_code = self.generate_expression(node.iterable)
            lines = [f'for (auto {var} : {iterable_code}) {{']
            
            self.indent_level += 1
            for stmt in node.body:
                stmt_code = self.generate_statement_code(stmt)
                if stmt_code:
                    lines.append(f'{self.indent()}{stmt_code}')
            self.indent_level -= 1
            
            lines.append(f'{self.indent()}}}')
            return '\n'.join(lines)
            
        elif isinstance(node, CountLoop):
            var = node.variable
            start_code = self.generate_expression(node.start)
            end_code = self.generate_expression(node.end)
            lines = [f'for (auto {var} = {start_code}; {var} <= {end_code}; {var}++) {{']
            
            self.indent_level += 1
            for stmt in node.body:
                stmt_code = self.generate_statement_code(stmt)
                if stmt_code:
                    lines.append(f'{self.indent()}{stmt_code}')
            self.indent_level -= 1
            
            lines.append(f'{self.indent()}}}')
            return '\n'.join(lines)
            
        elif isinstance(node, FunctionCall):
            return self.generate_expression(node) + ';'
            
        return ""
        
    def generate_if_statement(self, node: IfStatement) -> str:
        """Generate if statement code"""
        condition_code = self.generate_expression(node.condition)
        lines = [f'if ({condition_code}) {{']
        
        self.indent_level += 1
        for stmt in node.then_body:
            stmt_code = self.generate_statement_code(stmt)
            if stmt_code:
                lines.append(f'{self.indent()}{stmt_code}')
        self.indent_level -= 1
        
        # Handle elif parts
        for elif_condition, elif_body in node.elif_parts:
            elif_condition_code = self.generate_expression(elif_condition)
            lines.append(f'{self.indent()}}} else if ({elif_condition_code}) {{')
            
            self.indent_level += 1
            for stmt in elif_body:
                stmt_code = self.generate_statement_code(stmt)
                if stmt_code:
                    lines.append(f'{self.indent()}{stmt_code}')
            self.indent_level -= 1
            
        # Handle else part
        if node.else_body:
            lines.append(f'{self.indent()}}} else {{')
            
            self.indent_level += 1
            for stmt in node.else_body:
                stmt_code = self.generate_statement_code(stmt)
                if stmt_code:
                    lines.append(f'{self.indent()}{stmt_code}')
            self.indent_level -= 1
            
        lines.append(f'{self.indent()}}}')
        return '\n'.join(lines)
        
    def generate_expression(self, node: ASTNode) -> str:
        """Generate expression code"""
        if isinstance(node, Literal):
            if node.type == 'string':
                return f'"{node.value}"'
            else:
                return str(node.value)
                
        elif isinstance(node, Identifier):
            return node.name
            
        elif isinstance(node, BinaryOp):
            left_code = self.generate_expression(node.left)
            right_code = self.generate_expression(node.right)
            
            op_map = {
                TokenType.PLUS: '+',
                TokenType.MINUS: '-',
                TokenType.MULTIPLY: '*',
                TokenType.DIVIDE: '/',
                TokenType.MODULO: '%',
                TokenType.EQUALS: '==',
                TokenType.NOT_EQUALS: '!=',
                TokenType.LESS_THAN: '<',
                TokenType.GREATER_THAN: '>',
                TokenType.LESS_EQUAL: '<=',
                TokenType.GREATER_EQUAL: '>=',
                TokenType.AND: '&&',
                TokenType.OR: '||',
            }
            
            op = op_map.get(node.operator, '?')
            return f'({left_code} {op} {right_code})'
            
        elif isinstance(node, UnaryOp):
            operand_code = self.generate_expression(node.operand)
            
            if node.operator == TokenType.NOT:
                return f'!{operand_code}'
            elif node.operator == TokenType.MINUS:
                return f'-{operand_code}'
                
        elif isinstance(node, FunctionCall):
            args_code = ', '.join(self.generate_expression(arg) for arg in node.arguments)
            
            if node.name.startswith('new '):
                class_name = node.name[4:]  # Remove 'new '
                return f'make_shared<{class_name}>({args_code})'
            else:
                return f'{node.name}({args_code})'
                
        elif isinstance(node, MethodCall):
            object_code = self.generate_expression(node.object)
            
            if node.method == '[]':
                # Array access
                index_code = self.generate_expression(node.arguments[0])
                return f'{object_code}[{index_code}]'
            else:
                args_code = ', '.join(self.generate_expression(arg) for arg in node.arguments)
                if args_code:
                    return f'{object_code}.{node.method}({args_code})'
                else:
                    return f'{object_code}.{node.method}()'
                    
        elif isinstance(node, ArrayLiteral):
            if not node.elements:
                return 'vector<double>()'  # Empty vector constructor
            elements_code = ', '.join(self.generate_expression(elem) for elem in node.elements)
            return f'{{{elements_code}}}'
            
        elif isinstance(node, DictLiteral):
            pairs_code = []
            for key, value in node.pairs:
                key_code = self.generate_expression(key)
                value_code = self.generate_expression(value)
                pairs_code.append(f'{{{key_code}, {value_code}}}')
            return f'{{{", ".join(pairs_code)}}}'
            
        return "/* unknown expression */"
        
    def build_output(self) -> str:
        """Build final output"""
        lines = []
        
        # Add includes
        standard_includes = ['#include <iostream>', '#include <string>', '#include <vector>']
        for inc in standard_includes:
            if inc not in self.includes:
                self.includes.add(inc)
                
        if self.classes or any('map' in inc for inc in self.includes):
            self.includes.add('#include <map>')
            self.includes.add('#include <any>')
            
        if any('make_shared' in func for func in self.functions):
            self.includes.add('#include <memory>')
            
        for include in sorted(self.includes):
            lines.append(include)
            
        lines.append('')
        
        if self.output_type == "hpp":
            # Header guard
            guard_name = "SIMPLE_GENERATED_HPP"
            lines.insert(0, f'#ifndef {guard_name}')
            lines.insert(1, f'#define {guard_name}')
            lines.insert(2, '')
            
        # Add classes
        for class_def in self.classes:
            lines.append(class_def)
            lines.append('')
            
        # Add global variables
        for var in self.global_vars:
            if self.output_type == "hpp":
                lines.append(f'extern {var}')  # Declare as extern in header
            else:
                lines.append(var)
        if self.global_vars:
            lines.append('')
            
        # Add forward declarations
        for decl in self.forward_declarations:
            lines.append(decl)
        if self.forward_declarations:
            lines.append('')
            
        # Add functions
        for func in self.functions:
            lines.append(func)
            lines.append('')
            
        if self.output_type == "hpp":
            lines.append(f'#endif // {guard_name}')
            
        return '\n'.join(lines).rstrip() + '\n'
    
    def handle_string_interpolation(self, text: str) -> List[str]:
        """Handle string interpolation {variable} syntax"""
        parts = []
        current = ""
        i = 0
        
        while i < len(text):
            if text[i] == '{':
                # Found start of interpolation
                if current:
                    parts.append(f'"{current}"')
                    current = ""
                
                # Find end of interpolation
                i += 1
                var_name = ""
                while i < len(text) and text[i] != '}':
                    var_name += text[i]
                    i += 1
                
                if i < len(text) and text[i] == '}':
                    parts.append(var_name.strip())
                    i += 1
                else:
                    # Malformed interpolation, treat as literal
                    current += '{' + var_name
            else:
                current += text[i]
                i += 1
        
        if current:
            parts.append(f'"{current}"')
        
        return parts if parts else ['""']

# ============================================================================
# MAIN TRANSPILER CLASS
# ============================================================================

class SimpleTranspilerAST:
    """Main transpiler class with AST support"""
    
    def __init__(self):
        pass
        
    def transpile_file(self, input_file: str, output_file: str):
        """Transpile a .simple file to C++ or HPP"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Determine output type from file extension
            output_type = "hpp" if output_file.lower().endswith('.hpp') else "cpp"
            
            # Tokenize
            lexer = SimpleLexer(source_code)
            tokens = lexer.tokenize()
            
            # Parse
            parser = SimpleParser(tokens)
            ast = parser.parse()
            
            # Generate code
            generator = CodeGenerator(output_type)
            cpp_code = generator.generate(ast)
            
            # Write output
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cpp_code)
                
            print(f"Successfully transpiled {input_file} to {output_file}")
            
        except SimpleError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python simple_transpiler_ast.py input.simple output.cpp")
        print("       python simple_transpiler_ast.py input.simple output.hpp")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    transpiler = SimpleTranspilerAST()
    transpiler.transpile_file(input_file, output_file)

if __name__ == "__main__":
    main()