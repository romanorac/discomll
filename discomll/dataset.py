from disco.worker.task_io import task_input_stream ,chain_reader, gzip_line_reader
import urllib
import itertools

class Data:
	"""
	class Data initializes object with parameters that are needed for reading a dataset
	"""

	def __init__(self, data_tag, X_indices, **kvargs):
		"""
		Basic parameters
    	----------------
    	data_tag: list of strings - datatags on ddfs or urls on file server. Example, data_tag = ["test_data"].

		X_indices: list of integers - feature selection. Example, X_indices = range(1,4)
    	
    	X_meta: list of "c"s or "d" - define type for every feature (c - continuous, d - discrete). Example, X_meta = ["c","c","d"].

    	X_meta: string - defines a path to a file with features meta data. Example, X_meta = "/Users/home/user/metadata.txt" 
    	
    	id_index: integer - set index of sample identifier (some algorithms required to be set). Example, id_index = 0.
    	
    	y_index: integer - set label index. If not defined last index is used. Example, y_index = 4.
    	
    	data_type: string - define format of data. If not defined, it assumes that data is in text format. Otherwise, set "chunk" for chunked data on ddfs and "gzip" for gziped data. Example, data_type = "chunk".
    	
    	delimiter: string - set delimiter to parse the data. Example, delimiter = [","].


    	Other parameters
    	----------------
    	missing_vals: list of strings - define a missing value symbol. Example, missing_vals = ["?"]. 

    	y_map: list of two strings - parameter is used with SVM and logistic regression to map labels from string to integers. Example, y_map = ["red", "blue"].
    	
    	generate_urls: boolean - if True, it takes first and second url from data_tag list and generates all intermediate urls. If first filename is xaaaaa and second xaaaac, it will automatically generate url for xaaaab. It only works for split commands outputs that are named in x**** fashion. Example, generate_urls = True and data_tag = [[http://someurl/xaaaaa], [http://someurl/xaaazz]].
		"""
	   
		self.params = dict((k,v) for k, v in kvargs.iteritems()) #put parameters in a dictionary
		
		self.params["data_tag"] = data_tag #set data_tag
		if isinstance(self.params["data_tag"], basestring): #parameter data_tag should be in list
			self.params["data_tag"] = [self.params["data_tag"]]
		
		if self.params["data_tag"] == [""]: #Empty string raises Exception
			raise Exception("Data source should be defined.")
		
		if "generate_urls" in self.params and self.params["generate_urls"]:
			if len(data_tag) == 2:
				self.params["data_tag"] = self.generate_urls(data_tag[0], data_tag[1])
			else:
				raise Exception("Data tag should have first and last url defined if generate_urls parameter is set to true.")


		#set data_type of data_tag and define reader
		if "data_type" not in self.params or self.params["data_type"] == "" or self.params["data_type"] == "default" : 
			self.params["data_type"] = "default"
			self.params["input_chain"] = None
		elif self.params["data_type"] == "chunk": #data is chunked on ddfs
			self.params["input_chain"] = [task_input_stream, chain_reader] #reader for internal disco format
		elif self.params["data_type"] == "gzip": #data is in gziped form
			self.params["input_chain"] = [task_input_stream, gzip_line_reader] #reader for gziped data
		else:
			raise Exception("Parameter data_type should be undefined or have value chunk or gzip.")
	   
		self.params["X_indices"] = X_indices #Set features indices
		#Features indices should be defined in a list of integers or with xrange
		if not any(isinstance(self.params["X_indices"], t) for t in [list, xrange]) or not all(isinstance(item, int) for item in self.params["X_indices"]) or len(self.params["X_indices"]) == 0:
			raise Exception("X_indices should be list of integers.")
		
		#if id_index is not specified by user
		if "id_index" not in self.params or self.params["id_index"] == "": 
			self.params["id_index"] = -1 #dummy value
		#if value is string and if it it integer            
		elif isinstance(self.params["id_index"], basestring) and not self.params["id_index"].isdigit(): 
			raise Exception("Parameter id_index should be integer.")
		else:
			self.params["id_index"] = int(self.params["id_index"]) #set id_index
		
		#target index is not defined by user
		if "y_index" not in self.params or self.params["y_index"] == "": 
			self.params["y_index"] = -1 #last feature is used as target label in fitting phase. In prediction phase target label is not outputted if it is set to -1.
		#if value is string and if it it integer
		elif isinstance(self.params["y_index"], basestring) and not self.params["y_index"].isdigit(): 
			raise Exception("Parameter y_index should be integer.")
		else:
			self.params["y_index"] = int(self.params["y_index"]) #set y_index

		#delimiter is not set by user
		if "delimiter" not in self.params or self.params["delimiter"] == "": 
			self.params["delimiter"] = "," #default delimiter is comma.

		#Check if parameter "missing_vals" is defined
		if "missing_vals" not in self.params or self.params["missing_vals"] == "": 
			self.params["missing_vals"] = [] #no missing values
		#missing_vals param is passed as string
		elif isinstance(self.params["missing_vals"], basestring): 
			self.params["missing_vals"] = set(self.params["missing_vals"].split(",")) #missing values symbols are separated with comma.

		#no transformation of binary target variable is set
		if "y_map" not in self.params or self.params["y_map"] == "": 
			self.params["y_map"] = [] #default transformation ["True", "False"] 
		#y_map param is passed as string
		elif isinstance(self.params["y_map"], basestring):
			self.params["y_map"] = self.params["y_map"].replace(" ", "").split(",") #transformation values are separated with comma.
		elif len(self.params["y_map"]) != 2: #if there are more than 2 values defined for binary target
			raise Exception("Parameter y_map should have 2 values.")

		if "X_names" in self.params:
			if isinstance(self.params["X_names"], list):
				self.params["X_names"] = self.params["X_names"]
			else:
				raise Exception("Parameter X_names should be defined like X_names = [\"atr1\", \"atr2\"]")

			if len(X_indices) != len(self.params["X_names"]):
				raise Exception("Length of X_indices and X_names is not the same.")
		else:
			self.params["X_names"] = []

		if "X_meta" in self.params:
			if isinstance(self.params["X_meta"], basestring):
				f = urllib.urlopen(self.params["X_meta"])
				
				feature_names = f.readline().strip().split(self.params["delimiter"])
				feature_types = f.readline().strip().replace(" ", "").split(self.params["delimiter"])
				
				if feature_types == [""]:
					self.params["X_meta"] = feature_names
				else:
					self.params["X_meta"] = feature_types
					self.params["X_names"] = feature_names
					if len(feature_types) != len(feature_names):
						raise Exception("Define the same number of feature names as feature types.")
				
			elif isinstance(self.params["X_meta"], list):
				self.params["X_meta"] = self.params["X_meta"]
			if len(X_indices) != len(self.params["X_meta"]):
				raise Exception("Length of X_indices and meta is not the same.")
			if not set(self.params["X_meta"]).issubset(set(("c","d"))):
				raise Exception("Meta info should contain just c and d sings.")
		else:
			self.params["X_meta"] = []





		#If string parameters have unicode encoding we change it, to not interfere in algorithms
		temp_params = {}
		for k,v in self.params.iteritems():
			if isinstance(v, basestring):
				temp_params[str(k)] = str(v)
			else:
				temp_params[str(k)] = v
		self.params = temp_params
		
	def generate_urls(self, first_url, last_url):
		"""
		Function generates URLs in split command fashion. If first_url is xaaaaa and last_url is xaaaac, it will automatically generate xaaaab.
		"""
		first_url = first_url.split("/")
		last_url = last_url.split("/")
		if first_url[0].lower() != "http:" or last_url[0].lower() != "http:":
			raise Exception("URLs should be accessible via HTTP.")

		url_base = "/".join(first_url[:-1])
		start_index = first_url[-1].index("a")
		file_name = first_url[-1][0:start_index]
		url_base += "/" + file_name

		start = first_url[-1][start_index:]
		finish = last_url[-1][start_index:]

		file_extension = ""
		if start.count(".") == 1 and finish.count(".") == 1:
			start,file_extension = start.split(".")
			finish, _  = finish.split(".")
			if  len(start) != len(finish):
				raise Exception("Filenames in url should have the same length.")
			file_extension = "."+ file_extension
		else:
			raise Exception("URLs does not have the same pattern.")

		alphabet = "abcdefghijklmnopqrstuvwxyz"
		product = itertools.product(alphabet, repeat=len(start))

		urls = []
		for p in product:
			urls.append([url_base + "".join(p) + file_extension])
			if "".join(p) == finish:
				break			
		return urls

			

