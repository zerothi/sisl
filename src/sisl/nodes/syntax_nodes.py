from .node import Node


class SyntaxNode(Node):
    ...

class ListSyntaxNode(SyntaxNode):
    
    @staticmethod
    def function(*items):
        return list(items)
    

class TupleSyntaxNode(SyntaxNode):
    
    @staticmethod
    def function(*items):
        return tuple(items)
    
class DictSyntaxNode(SyntaxNode):
    
    @staticmethod
    def function(**items):
        return items