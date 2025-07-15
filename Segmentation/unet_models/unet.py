
import segmentation_models_pytorch as smp
#def get_unet():



    
#    return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)


def get_unet(in_channels=3):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1
    )
