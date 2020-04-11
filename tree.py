class Node(object):
	def __init__(self, data):
		self.data = data
		self.children = set()

	def add_child(self, obj):
		self.children.add(obj)

	def remove_child(self, obj):
		if obj in self.children:
			self.children.remove(obj)
		else:
			for child in self.children:
				if child.data == obj.data:
					self.children.remove(child)
					return

def make_tree(root, graph):
	node = Node(root)
	for child in graph[root]['neighbors']:
		node.add_child(make_tree(child, graph))
	return node

def print_tree(root):
	print(root.data)
	for child in root.children:
		print_tree(child)

def main():
	x = {0 : [1, 3], 1 : [2], 2: [4, 5], 3 : [], 4 : [6], 5: [], 6: []}
	tree = make_tree(0, x)
	print_tree(tree)


if __name__ == '__main__':
	main()