 # Part 6: Image KNN Classifier #
 class ImageKNNClassifier:
     """
     Represents a simple KNNClassifier
     """

     def __init__(self, k_neighbors):
         """
         Creates a new KNN classifier object
         """
         
         self.k_neighbors = k_neighbors
         
         # List of (image, label) tuples where 'image' is a RGBImage and 'label' is a string
         self.data = []

     def fit(self, data):
         """
         Stores the given set of data and labels for later
         """
         
         assert isinstance(data, list)
         assert all(isinstance(item, tuple) and len(item) == 2 for item in data)
         assert all(isinstance(item[1], str) for item in data)
         
         if len(data) < self.k_neighbors:
            raise ValueError
        
        self.data = data

     def distance(self, image1, image2):
         """
         Returns the distance between the given images

         >>> img1 = img_read_helper('img/steve.png')
         >>> img2 = img_read_helper('img/knn_test_img.png')
         >>> knn = ImageKNNClassifier(3)
         >>> knn.distance(img1, img2)
         15946.312896716909
         """
         
         # Parameters
         # - Image1: RGBImage instance
         # - Image2: RGBImage instance
         
         # Exceptions
         # - If either (or both) are not RGBImage instances, raise a TypeError().
         # - If they are not the same size, raise a ValueError().
         
         if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
            
         if image1.size != image2.size:
            raise ValueError
            
         flat_image1 = [pixel for row in image1.pixels for pixels in row]
         flat_image2 = [pixel for row in image2.pixels for pixels in row]
        
         sum_squared_difference = sum(map(lambda p1, p2: (p1 - p2) ** 2, flat_image1, flat_image2))
        
         euclidean_distance = sum_squared_difference ** 0.5
        
         return euclidean_distance

     def vote(self, candidates):
         """
         Returns the most frequent label in the given list

         >>> knn = ImageKNNClassifier(3)
         >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
         'label2'
         """
         
         count = 0
         most_frequent_label = None
         
         for label in candidates:
            temp_count = 0
            
            for other_label in candidates:
                if label == other_label:
                    temp_count += 1
            
            if temp_count > count:
                count = temp_count
                most_frequent_label = label
        
         return most_frequent_label

     def predict(self, image):
         """
         Predicts the label of the given image using the labels of
         the K closest neighbors to this image

         The test for this method is located in the knn_tests method below
         """
         # YOUR CODE GOES HERE #


 def knn_tests(test_img_path):
     """
     Function to run knn tests

     >>> knn_tests('img/knn_test_img.png')
     'nighttime'
     """
     # Read all of the sub-folder names in the knn_data folder
     # These will be treated as labels
     path = 'knn_data'
     data = []
     for label in os.listdir(path):
         label_path = os.path.join(path, label)
         # Ignore non-folder items
         if not os.path.isdir(label_path):
             continue
         # Read in each image in the sub-folder
         for img_file in os.listdir(label_path):
             train_img_path = os.path.join(label_path, img_file)
             img = img_read_helper(train_img_path)
             # Add the image object and the label to the dataset
             data.append((img, label))

     # Create a KNN-classifier using the dataset
     knn = ImageKNNClassifier(5)

     # Train the classifier by providing the dataset
     knn.fit(data)

     # Create an RGBImage object of the tested image
     test_img = img_read_helper(test_img_path)

     # Return the KNN's prediction
     predicted_label = knn.predict(test_img)
     return predicted_label
