def ocr_engines():
    # Cache the engines after the first import, so subsequent calls are fast
    if not hasattr(ocr_engines, "_cached"):
        from docling.models.easyocr_model import EasyOcrModel
        from docling.models.ocr_mac_model import OcrMacModel
        from docling.models.rapid_ocr_model import RapidOcrModel
        from docling.models.tesseract_ocr_cli_model import TesseractOcrCliModel
        from docling.models.tesseract_ocr_model import TesseractOcrModel

        ocr_engines._cached = {
            "ocr_engines": [
                EasyOcrModel,
                OcrMacModel,
                RapidOcrModel,
                TesseractOcrModel,
                TesseractOcrCliModel,
            ]
        }
    return ocr_engines._cached


def picture_description():
    from docling.models.picture_description_api_model import PictureDescriptionApiModel
    from docling.models.picture_description_vlm_model import PictureDescriptionVlmModel

    return {
        "picture_description": [
            PictureDescriptionVlmModel,
            PictureDescriptionApiModel,
        ]
    }
