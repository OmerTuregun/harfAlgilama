<<<<<<< HEAD
# Preprocessing (Önişleme):
preprocess_frame fonksiyonu ile görüntü gri tonlamaya çevrilir, gürültü azaltmak için Gauss filtresi uygulanır.

# Hareket Algılama:
detect_hand_movements fonksiyonu ile el hareketlerini algılar. İlk olarak arka planı tespit eder ve hareket eden eli tespit eder. accumulateWeighted fonksiyonu ile arka plan modeli sürekli olarak güncellenir.

# Şekil Tanıma:
recognize_shapes fonksiyonu ile el üzerindeki şekilleri tanımlar. Bu örnekte sadece elin alanını (area) ve merkezini (center) hesaplar.

# Ana Program:
main fonksiyonunda webcam'den görüntü alınır, el hareketleri algılanır ve tanınan şekillerin bilgileri ekrana yazdırılır.
=======
# Harf-Alg-lama
WebCam aracılığıyla el hareketiyle yapılan harfleri algılayan makine öğrenmesine dayalı bir python projesi
>>>>>>> 8b276b87c443fbe7f859bcb4b4172092786dd90e
