import numpy as np
import math

def my_fit( words, verbose = False ):
	dt = Tree( min_leaf_size = 1, max_depth = 15 )
	dt.fit( words, verbose )
	return dt


class Tree:
	def __init__( self, min_leaf_size, max_depth ):
		self.root = None
		self.words = None
		self.min_leaf_size = min_leaf_size
		self.max_depth = max_depth
	
	def fit( self, words, verbose = False ):
		self.words = words
		self.root = Node( depth = 0, parent = None )
		if verbose:
			print( "root" )
			print( "└───", end = '' )
		# The root is trained with all the words
		self.root.fit( all_words = self.words, my_words_idx = np.arange( len( self.words ) ), min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, verbose = verbose )


class Node:
	def __init__( self, depth, parent ):
		self.depth = depth
		self.parent = parent
		self.all_words = None
		self.my_words_idx = None
		self.children = {}
		self.is_leaf = True
		self.query_idx = None
		self.history = []
	
	def get_query( self ):
		return self.query_idx
	
	def get_child( self, response ):
		if self.is_leaf:
			print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
			child = self
		else:
			if response not in self.children:
				print( f"Unknown response {response} -- need to fix the model" )
				response = list(self.children.keys())[0]
			
			child = self.children[ response ]
			
		return child
	
	def process_leaf( self, my_words_idx, history ):
		return my_words_idx[0]
	
	def reveal( self, word, query ):
		mask = [ *( '_' * len( word ) ) ]
		
		for i in range( min( len( word ), len( query ) ) ):
			if word[i] == query[i]:
				mask[i] = word[i]
		
		return ' '.join( mask )
	
	def process_node( self, all_words, my_words_idx, history, verbose ):
		split_dict = {}
		if len( history ) == 0:
			query_idx = -1
			query = ""
		else:
			query_idx = self.get_query_idx(all_words, my_words_idx)
			query = all_words[ query_idx ]

			
		split_dict = self.split(all_words, my_words_idx, query)
		
		if len( split_dict.items() ) < 2 and verbose:
			print( "Warning: did not make any meaningful split with this query!" )

		return ( query_idx, split_dict )
	
	def fit( self, all_words, my_words_idx, min_leaf_size, max_depth, fmt_str = "    ", verbose = False ):
		self.all_words = all_words
		self.my_words_idx = my_words_idx
		
		if len( my_words_idx ) <= min_leaf_size or self.depth >= max_depth:
			self.is_leaf = True
			self.query_idx = self.process_leaf( self.my_words_idx, self.history )
			if verbose:
				print( '█' )
		else:
			self.is_leaf = False
			( self.query_idx, split_dict ) = self.process_node( self.all_words, self.my_words_idx, self.history, verbose )
			
			if verbose:
				print( all_words[ self.query_idx ] )
			
			for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
				if verbose:
					if i == len( split_dict ) - 1:
						print( fmt_str + "└───", end = '' )
						fmt_str += "    "
					else:
						print( fmt_str + "├───", end = '' )
						fmt_str += "│   "
				
				self.children[ response ] = Node( depth = self.depth + 1, parent = self )
				history = self.history.copy()
				history.append( [ self.query_idx, response ] )
				self.children[ response ].history = history
				
				self.children[ response ].fit( self.all_words, split, min_leaf_size, max_depth, fmt_str, verbose )		
	
	def split(self, all_words, my_words_idx, query):
		split_dict = {}
		
		for idx in my_words_idx:
			mask = self.reveal( all_words[ idx ], query )
			if mask not in split_dict:
				split_dict[ mask ] = []
			
			split_dict[ mask ].append( idx )
		return split_dict
  
	def get_query_idx( self, all_words, my_words_idx ):
		max_entropy = 0
		query_idx = -1
		for idx in my_words_idx:
			new_split = self.split(all_words, my_words_idx, all_words[idx])
			entropy = self.compute_entropy(new_split)
			if entropy > max_entropy:
				max_entropy = entropy
				query_idx = idx
		
		return query_idx
			

	def compute_entropy( self, split ):
		entropy = 0.0
		total_words = len(self.my_words_idx)
		for s in split:
			p = len(s) / total_words
			entropy -= p * math.log(p)
		return entropy