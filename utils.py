import SimpleITK as sitk


def copy_image_info(image1, image2):
    if not isinstance(image1, sitk.Image) or not isinstance(image2, sitk.Image):
        raise ValueError("Both inputs must be SimpleITK Image objects.")

    image2.CopyInformation(image1)

    return image2
