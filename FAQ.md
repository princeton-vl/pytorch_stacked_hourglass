Q: What exactly are the differences in parameters between this code and the Stacked Hourglass paper? <br/>
A: The paper uses batch size of 8 instead of 16, and RMSprop with learning rate 2.5e-4 instead of Adam with learning rate of 1e-3. Also they decayed learning rate "after validation accuracy plateaued" instead of explicity at 100k iterations, but this is more or less the same idea. You can change Adam to RMSProp in "make_network" function within task/pose.py, while learning rate and batch size are in task/pose.py.

Q: Were scores on 2HG model achieved using same parameters as 8HG? <br/>
A: Yes, just change nstack to 2 in task/pose.py

Q: How do I interpret the output of the log file? <br/>
A: Each iteration during training or validation outputs a line to this file with corresponding loss. Note, we do not calculate train or validation accuracy during training as this operation requires preprocessing and is expensive. Validation loss can be used as a proxy for when to stop training.

Q: Only one model is saved during training? <br/>
A: Yes - the most recent model is saved each epoch. You may want to modify if you desire to save "best" checkpoint, etc.

Q: How can I display predictions? <br/>
A: There isn't explicit visualization code here, but you can change pixels of the image corresponding to keypoints to visualize. To do this, for example, you could modify mpii_eval function in test.py to take in images as well as keypoints and write this to tensorboard.

Q: The evaluation code evaluates train and validation set? What about the test set? <br/>
A: Yes, the evaluation code (test.py) is setup to calculate accuracy on the validation set. Train accuracy (like validation accuracy) is not calculated during training, so is also calculated here. Default settings in task/pose.py are setup to calculate train accuracy on a sample of the train set (300 images) to reduce compute time. To get test accuracy, you must run test.py with images and crops from test.h5, then use the evaluation toolkit provided by MPII, and submit as detailed on the [MPII website](http://human-pose.mpi-inf.mpg.de/#evaluation).
