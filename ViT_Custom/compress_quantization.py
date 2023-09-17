from ViT_Custom.ViT_Custom import ViT
from performance_evaluator import eval_model
#script to quantize the model using Post Training Quantization (PTQ)



#function to quantize the parameter of the model after training
def quantization(model: ViT):
    return model




model = ViT()

#load weights from pretrained model


#quantize the model, then test the differences
quant_model = quantization(model)

