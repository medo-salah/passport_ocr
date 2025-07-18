from pyzbar import pyzbar

def extract_barcodes(image):
    barcodes = pyzbar.decode(image)
    decoded = [barcode.data.decode("utf-8") for barcode in barcodes]
    return decoded