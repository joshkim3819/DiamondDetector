# DiamondDetector

Purpose:  Create a model that can determine the authenticity of a diamond given the validation images with 80% accuracy

Inspiration: Papers on YOLOv5 and EfficientNetB3

WARNING: Did not succeed due to 3 main reasons
  1. **Insufficient Accurate data** - Cannot determine if internet images are fake diamonds
  2. **Diamonds have different ratings** - Some authentic diamonds have a yellowish hue due to Nitrogen in the diamond formation
  3. **Variable Lighting** - Diamond's sparkles vary heavily on how and what kind of light is being used

  
# Model Architecture (Script found in "Detection")

The model is composed of three main steps
  1. **Detection**: Utilized Pytorch's YOLOv5 to identify and label diamonds within provided images (Images would ideally have only diamonds, but could include diamonds on rings or necklaces)
  3. **Classification**: Utilize Tensorflow's EfficientNetB3 to finetune model to assist in categorizing real/fake diamonds
  4. **Evaluation**: Score of how well the model performed (correct/total sample size)
  5. **Data**:
    a. Training Data: Includes Real and Fake
    b. Validation Data: Includes Real and Fake 
