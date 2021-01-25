# Classifying Hip-Hop Tracks By Regional Style Using Computer Vision and Natural Langauge Processing


## Overview

I love hip-hop.

There's something about the headnodic beats and melodic exercise of word-smithing that furrows the brow and contorts the face in an expression of anguish that can only be acheived through a harmony of beats, rhymes, and life.

You don't need to be a fan of hip-hop to experience the power of music. Stevie Wonder himself reminds us that it's, "... a language we all understand." I happen to just have particular affinity for hip-hop, and I like many others, am able to discern where an artists comes from based on their style.

This project is a journey into classifying hip hop based on regional styles using computer vision and natural language processing. 

## Business Problem

The goal of this project is to be able to classify music by regional style to then be used by streaming services or Digital Service Provider's (DSP's) to create more accurate sub-genre playlists that will better cater to their target audience. 

## Data

The data consists of over 9000 lyrics and other features gathered from [Genius](genius.com) and my personal collection of artists discographies. After removing mix-tapes, and other tracks such as intros, outros, interludes, and skits, I ended up with almost 4000 tracks.

ADD PICTURES OF SPECTORGRAMS


## Results

ADD PICTURES OF CONFUSION MATRICES
![report](./images/report.png)

After several iterations, our best NLP model was a random forest with a 73% accuracy rate of 95% while our computer vision portion was a CNN with an accuracy of 53%.

I would like to try again using the pretrained model BERT to see if I can increase the accuracy for my NLP model. For the computer vision side, I think using the pretrained model ResNet34 would gain considerable improvements, but before that I would like to see how the results are affected by adding more data. I don't believe that around 1000 images for each of the 4 regions seems like quite enough, and I think my models would benefit from additional data. Another thing I would like to do is identify which tracks have featured artists. Because the entire song is converted into an image, there are songs feature artists from different regions on the same track which I believe skews my results.

## Recommendations

This analysis leads to four recommendations should you work to attain or improve upon the results of these models:

 * More data for CNN
   * While there were about 4 thousands images total given within the dataset. I believe that better results could be gathered if there was more data for the model to train on. CNN's will always benefit from more data as long as there are enough resources available to process it effectively.
 * Transfer Learning
   * Using transfer learning from a model that was pre-trained such as ResNet34 and BERT. Fine tuning will take considerably less time and resources and yield just as good if not better results from the right model.
 * Remove Featured Artists
   * Before converting files into spectograms, be more meticulous in scrubbing the data remove tracks featuring artists from other regions.
 * Clip Audio
   * Create spectograms containing only 45 seconds of audio data starting from 0:30 to 1:15 in order to reduce the resources needed to train your CNN. This time frame is likely to include most of the first verse and chorus which should be consistent to the rest of the tracks regional style.


### Next Steps

Given more time I would look forward to implementing the following:

* Model By Era
    * Segment predictions by era instead of datasets that encompass hip-hop from all time to see how it becomes more difficult as the genre grows.
* Clustering
    * Use a clustering method allowing the model to identify styles and cross reference that with our own domain knowledge.
* Map Visualization
    * Create a map to visualize the regions each artist and their tracks are from and compare it to where the the model predicts that it is from to discern patterns.
