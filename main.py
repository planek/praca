import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Lemon.jpg/500px-Lemon.jpg"

def download_image_from_url(url):
    """
    Pobiera zdalne zdjęcie i zwraca obiekt Image z Pillow.
    Zawiera nagłówek User-Agent, aby uniknąć błędu 403.
    """
    print(f"Pobieranie zdjęcia z: {url}")
    
 
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        return Image.open(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Błąd podczas pobierania zdjęcia: {e}")
        return None

def calculate_and_plot_histogram(img, title="Histogram zdjęcia"):
    """
    Oblicza i rysuje histogramy dla skali szarości oraz kanałów R, G, B.
    """
    img_gray = img.convert('L')
    hist_gray = np.array(img_gray.histogram())
    
    r, g, b = img.split()
    hist_r = np.array(r.histogram())
    hist_g = np.array(g.histogram())
    hist_b = np.array(b.histogram())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    def plot_hist(ax, hist_data, color, label):
        ax.bar(range(256), hist_data, color=color, alpha=0.9, width=1.0)
        ax.set_title(label)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Wartość piksela (0-255)")
        ax.set_ylabel("Liczba pikseli")

    plot_hist(axes[0, 0], hist_gray, 'gray', 'Całe zdjęcie (Skala szarości)')
    plot_hist(axes[0, 1], hist_r, 'red', 'Kanał Czerwony (R)')
    plot_hist(axes[1, 0], hist_g, 'green', 'Kanał Zielony (G)')
    plot_hist(axes[1, 1], hist_b, 'blue', 'Kanał Niebieski (B)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return hist_gray

def estimate_quality(hist_gray):
    """
    Szacuje jakość zdjęcia na podstawie właściwości histogramu skali szarości.
    """
    total_pixels = np.sum(hist_gray)

    SHADOW_CLIP_THRESHOLD = 0.005 
    HIGHLIGHT_CLIP_THRESHOLD = 0.005

    shadow_clip_count = np.sum(hist_gray[0:3]) 
    highlight_clip_count = np.sum(hist_gray[253:256]) 

    is_shadow_clipped = shadow_clip_count > (total_pixels * SHADOW_CLIP_THRESHOLD)
    is_highlight_clipped = highlight_clip_count > (total_pixels * HIGHLIGHT_CLIP_THRESHOLD)
    
    cumulative_hist = np.cumsum(hist_gray)

    try:
        min_val = np.where(cumulative_hist > total_pixels * 0.01)[0][0]
        max_val = np.where(cumulative_hist > total_pixels * 0.99)[0][0]
    except IndexError:
        min_val = 0
        max_val = 255

    tonal_range = max_val - min_val
    MIN_TONAL_RANGE = 180

    quality_score = {
        "ocena": "Dobra",
        "opis": "Histogram jest dobrze rozłożony. Zdjęcie ma zadowalający kontrast.",
        "min_val": min_val,
        "max_val": max_val,
        "tonal_range": tonal_range,
        "shadow_clip_count": shadow_clip_count,
        "highlight_clip_count": highlight_clip_count
    }

    if is_shadow_clipped and is_highlight_clipped:
        quality_score["ocena"] = "Wadliwa - Ekstremalne Przycięcie"
        quality_score["opis"] = "Znaczne przycięcie w Cieniach i Światłach."
    elif is_shadow_clipped:
        quality_score["ocena"] = "Wadliwa - Przycięcie Cieni"
        quality_score["opis"] = "Znaczne przycięcie w Cieniach (zbyt ciemne)."
    elif is_highlight_clipped:
        quality_score["ocena"] = "Wadliwa - Przycięcie Świateł"
        quality_score["opis"] = "Znaczne przycięcie w Światłach (zbyt jasne)."
    elif tonal_range < MIN_TONAL_RANGE:
        quality_score["ocena"] = "Wadliwa - Niski Kontrast"
        quality_score["opis"] = f"Histogram jest zbyt wąski (zakres: {tonal_range})."
        
    return quality_score, min_val, max_val

def improve_quality(img, min_val, max_val):
    """
    Implementuje Rozciąganie Kontrastu (Normalizację).
    """
    img_array = np.array(img, dtype=np.float32)

    if max_val > min_val:
     
        img_array = np.clip(img_array, min_val, max_val)
      
        img_corrected_array = (img_array - min_val) * (255.0 / (max_val - min_val))
        
        img_corrected = Image.fromarray(img_corrected_array.astype(np.uint8))
        return img_corrected
    else:
        print("Nie można zastosować poprawy: Minimalna i maksymalna wartość są identyczne.")
        return img

if __name__ == "__main__":
    original_img = download_image_from_url(IMAGE_URL)

    if original_img:
        print("\n--- 1. Analiza Oryginalnego Zdjęcia ---")
        plt.figure(figsize=(6, 6))
        plt.title("Oryginalne Zdjęcie")
        plt.imshow(original_img)
        plt.axis('off')
        plt.show()

        hist_gray_original = calculate_and_plot_histogram(original_img, "Histogramy Oryginalnego Zdjęcia")

        quality_report, min_val, max_val = estimate_quality(hist_gray_original)
        
        print("\n--- 2. Szacowanie Jakości Zdjęcia (Algorytm) ---")
        print(f"OCENA: {quality_report['ocena']}")
        print(f"OPIS: {quality_report['opis']}")
        print(f"Wykryty Zakres Tonalny: {quality_report['min_val']} - {quality_report['max_val']} (Rozpiętość: {quality_report['tonal_range']})")
        
      
        print("\n--- 3. Bonus: Zastosowanie Algorytmu Poprawy (Rozciąganie Kontrastu) ---")
        corrected_img = improve_quality(original_img, min_val, max_val)
        
        plt.figure(figsize=(6, 6))
        plt.title("Zdjęcie Po Korekcie (Rozciąganie Kontrastu)")
        plt.imshow(corrected_img)
        plt.axis('off')
        plt.show()

        calculate_and_plot_histogram(corrected_img, "Histogramy Zdjęcia Po Korekcie")
        print("Gotowe.")
    else:
        print("Nie udało się pobrać zdjęcia, sprawdź URL lub połączenie internetowe.")