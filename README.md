# Disaster-Relief-Google-Girl-Hackathon
## Orchestrating Disaster Relief Efforts through Space and Social Media Analytics

This solution employs machine learning, satellite imagery, and social media analytics to enhance the targeting of disaster relief operations. The project's focus was on Typhoon Haiyan, a catastrophic event that struck the Philippines in November 2013, setting records for the highest wind speeds at landfall and resulting in the destruction of over a million homes.

In the aftermath of natural disasters, it's crucial to identify the areas that have sustained the most damage to effectively allocate relief resources. Damage assessment maps are typically produced by volunteers from the Humanitarian Open Street Map team, who manually label each building based on their evaluation of the damage, using comparisons of satellite imagery taken before and after the disaster. However, the creation of these maps is labor-intensive and time-consuming, and the results may not always be accurate. 

Conversely, social media often serves as a rich source of information about disasters, as individuals frequently share updates about their experiences, making it a valuable tool for accurately pinpointing affected areas. This method harnesses data from both social media and satellite imagery, applying machine learning algorithms to identify regions hit by disasters. The integration of these data sources can significantly enhance the efficiency and effectiveness of disaster relief efforts.


## Goal

The objective is to develop a model that can swiftly and accurately pinpoint the areas most severely affected by disasters, thereby optimizing the allocation of relief resources.

Utilizing satellite imagery from before and after Typhoon Haiyan's impact on the Philippines, I incorporated social media analytics and constructed a neural network to identify damaged buildings. The model's predictions were then used to generate density maps of the damage, highlighting the areas most in need of relief efforts.

The ultimate aim of this implementation is to establish a robust and efficient approach for extracting building footprints from high-resolution aerial images. This has significant practical implications in fields such as urban planning, geospatial database updates, and disaster management, where precise building information is vital for informed decision-making and analysis.


## Data

I utilized Landsat8 satellite imagery, freely accessible via Google Earth Engine, which offers a resolution of 15 meters. While this resolution is significantly lower than what is available commercially (up to 30 cm per pixel), the effectiveness of this model, even when using publicly available data, underscores its potential use for organizations with funding constraints.

<p align="center">
<img src="https://github.com/ejm714/disaster_relief_from_space/blob/master/imgs/satellite.png?raw=true" alt="Pre and post typhoon satellite imagery" width="600">
</p>

Above is an example of satellite imagery pre and post typhoon for Tacloban City, one of the hardest hit areas. On the left, lighter color squares representing buildings are visible. On the right, there is much more grey as these buildings were destroyed.

My ground truth data on building damage came from the Copernicus Emergency Management Service. 

<p align="center">
<img src="https://github.com/ejm714/disaster_relief_from_space/blob/master/imgs/buildings.png?raw=true" alt="Building damage" width="300">
</p>

An example of my building damage data is shown above, with different colors squares indicating different levels of damage. Superimposing these polygons on my satellite imagery, I labeled each pixel in my satellite imagery as either being part of a damaged building or not.


## Social Media Analytics 

I have provided the separate notebook 'Social_Media_Data_Integration.ipynb' for gathering the data from social media. This Python script uses the Tweepy library to fetch tweets from Twitter. It begins by importing Tweepy and setting up authentication with the Twitter API using unique keys and tokens associated with a Twitter developer account. An API object is created to interact with Twitter. The script then defines the search parameters, including the search query (hashtags related to Typhoon Haiyan) and the date from which to start collecting tweets. Using the Tweepy Cursor class, the script collects a specified number of tweets that match the search query and were posted since the specified date. Finally, it iterates over the collected tweets and prints the text of each tweet.


## Integrating the Social Media data with Satellite Data

After collecting the data from the social media the code then loads a satellite image of Typhoon Haiyan/Yolanda from a file. The image is converted to a numerical array using the numpy library.
Next, the code creates a figure with two subplots using the matplotlib.pyplot library. The first subplot is used to display the satellite image, and the second subplot is used to display the collected tweets.
In the first subplot, the code displays the satellite image using the imshow function and sets a title for the subplot.
In the second subplot, the code first sets a title and turns off the axis, as it will be displaying text (tweets) instead of numerical data. It then creates a list of tweet texts by extracting the text from each tweet object.
The code then iterates over the list of tweet texts and uses the text function to display each tweet in the second subplot. The text function allows for wrapping long tweets to multiple lines.
Finally, the code adjusts the layout of the figure using the tight_layout function and displays the figure using the show function from the matplotlib.pyplot library.


## Random forest baseline model

My baseline model was a random forest pixel-wise classification (with each pixel either being damaged or not damaged). The model features came from the band information from my satellite data, i.e., for each pixel, how much red it has, how much blue, green, UV, infrared, etc.

I did a pixel wise subtraction between pre-typhoon and post-typhoon imagery given that it is a change detection model. I wanted to understand whether there was a building there before but isn't there anymore.

However many things can change — buildings can collapse and vegetation can bloom. Yet vegetation reflects infrared really highly whereas buildings do not. By including the pre and post pixel values in addition to the subtraction values, I was able to better identify buildings.

To deal with a class imbalance, I trained the model on a balanced sample. This made it more sensitive to damaged buildings.

When I trained a random forest on the top half of three satellite images and had it predict on the bottom half, it did very well as shown by the ROC curve which is very close to the upper left hand corner. However when I asked the model to predict on an image it hasn't seen before, the ROC curve dropped precipitously. This indicates that the model does a poor job of generalizing. The model is therefore hard to scale as it would be infeasible to train a model on every part of a country after a natural disaster.

![alt text](https://github.com/ejm714/disaster_relief_from_space/blob/master/imgs/random_forest.png?raw=true "Random forest ROC curves")

One reason why the model struggles to generalize may be that satellite images can be taken at different times of day and therefore can have different lighting (note the differences below) — meaning that the thresholds identified in one model may not apply to other imagery.

<p align="center">
<img src="https://github.com/ejm714/disaster_relief_from_space/blob/master/imgs/light_diffs.png?raw=true" alt="Satellite imagery comparison">
</p>


## Fully Connected Neural Network Model 

This Python script is designed to build, train, and evaluate a fully connected neural network for a binary classification task using the Keras library. It starts by defining the model architecture, which consists of an input layer with 32 neurons, a hidden layer with 16 neurons, and an output layer with 1 neuron (for binary classification). The model is then compiled with the binary cross-entropy loss function, the Adam optimizer, and the Area Under the Curve (AUC) as the metric.

The model is trained on the training data (X_train, y_train) for 50 epochs with a batch size of 32. After training, the model’s performance is evaluated on the test data (X_test, y_test). The model’s AUC score is printed out.

The model is then used to make predictions on the test data. Both class predictions (y_pred) and probability predictions (y_proba) are generated. The class predictions are used to print a classification report, which includes precision, recall, f1-score, and support for each class.


## A superior U-Net model

To create a model that is more generalizable, I built a neural net called a U-Net, named for its structure of convolutional layers. U-Nets are known to be good for object detection (segmentation).

This model did quite well with my data. The ROC on the left is for the validation set, which are images the model has seen. On the right is the ROC curve for the holdout set, which are images the model has never seen before. Unlike the random forest, the ROC curve didn't drop significantly. The fact that this model extends well to new images is important as one would want to feed in satellite imagery for an entire country to generate hotspot predictions after a natural disaster.

![alt text](https://github.com/ejm714/disaster_relief_from_space/blob/master/imgs/unet_roc.png?raw=true "U-Net ROC")


## Building Footprint Extraction from High Resolution Aerial Images Using Generative Adversarial Network (GAN) Architecture

The GAN architecture incorporates a SegNet model with Bi-directional Convolutional LSTM (BConvLSTM). The SegNet model is used to generate the segmentation map from the high-resolution aerial imagery dataset. BConvLSTM is then employed to combine encoded features (containing local information) and decoded features (containing semantic information), improving the model's performance in the presence of complex backgrounds and barriers. 

The proposed technique achieves promising results, with an average F1-score of 96.81%.


## Mapping the density of damage

The U-Net predictions identify where the damaged buildings are, and from this, I created a density map highlighting the areas with the greatest building damage.

![alt text](https://github.com/ejm714/disaster_relief_from_space/blob/master/imgs/ground_truth_and_prediction.png?raw=true "Ground truth and prediction")

On the left is the post-typhoon satellite imagery. In the middle is the ground truth data with the black marking damaged buildings — this aligns with the grey in the satellite imagery. On the right is a density map of damage based on predictions from the model with the darkest areas being the most damaged areas, and therefore priority areas for relief efforts. As the predictions are on a holdout set, the similarity between ground truth and prediction illustrates the model's ability to scale.


# I have provided the Output Images in the folder named 'Output Images'.


## Applications

This sort of change detection model has many applications. For example, 

1. Damage assessment through satellite image analysis.
2. Optimizing resource allocation based on real-time social media updates.
3. Aiding search and rescue operations by identifying areas with survivors.
4. Assisting in evacuation planning and identifying high-risk zones.
5. Supporting post-disaster recovery efforts by monitoring reconstruction progress.
6. Developing early warning systems for different types of disasters.

Overall, this integrated system enhances disaster management and relief operations by providing accurate information, optimizing resource allocation, and aiding decision-making processes.

-----
**Languages**: Python, JavaScript  
**Libraries**: Keras + TensorFlow, numpy, pandas, sklearn, rasterio, geopandas, shapely, opencv, matplotlib, seaborn  
**Methods**: Deep learning, classification (supervised learning), GAN Architecute, BConvLSTM, SegNet Model   

Replication notes:

- `google-earth-engine-satellite.js` pulls the satellite imagery from Google Earth Engine and should be run first.
- Building data shapefiles (grading maps) can be downloaded manually from the <a href="http://emergency.copernicus.eu/mapping/list-of-components/EMSR058">Copernicus Emergency Management Service website</a>.
- `Data_Preprocessing.ipynb` is a precursor to two modeling notebooks. 'Social_Media_Data_Integration.py' file is for gathering the social media data and then integrating them with the satellite data. The latter two can be run independently of one another.
