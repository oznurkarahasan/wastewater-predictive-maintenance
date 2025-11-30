# Atık Su Arıtma Tesislerinde Anomali Tespiti ve Arıza Tahmini (Predictive Maintenance)

## Proje Özeti

Bu proje, atık su arıtma tesislerinde kullanılan pompaların sensör verilerini analiz ederek olası arızaları önceden tahmin etmeyi (predictive maintenance) ve anomali tespiti yapmayı amaçlar.

Su yönetimi, çevresel sürdürülebilirlik ve halk sağlığı için kritik bir sektördür. Tesislerdeki beklenmedik pompa arızaları; çevre kirliliğine, yeraltı suyu hasarına ve yüksek bakım maliyetlerine yol açabilir. Bu proje, Makine Öğrenmesi (ML) teknikleri kullanarak bu riskleri minimize etmeyi hedefler.

## Problem Tanımı

Atık su arıtma tesisleri, suyun çevreye ve insan sağlığına zarar vermeden yeniden kullanılabilir hale getirilmesini sağlayan kritik altyapı sistemleridir. Bu tesislerde kullanılan pompalar, motorlar, sensörler, oksijen ölçüm cihazları ve kimyasal izleme ekipmanları zaman içinde yıpranma, kalibrasyon bozukluğu veya mekanik arızalar nedeniyle doğru çalışmamaya başlayabilir.

Bu tür bozulmalar erken tespit edilmediğinde:

- Arıtma verimliliği düşer,
- Arıtılmamış veya yetersiz arıtılmış su çevreye salınabilir,
- Yeraltı ve yüzey suları kirlenebilir,
- Enerji ve bakım maliyetleri artar,
- Beklenmedik ekipman arızaları nedeniyle tesis durabilir,
- İnsan sağlığı ciddi risk altına girer.

Bu nedenle, ekipman ve sensör performansını sürekli izlemek ve olası arızaları önceden öngörmek kritik bir ihtiyaçtır.

Bu probleme çözüm olarak; tarihsel sensör verilerine dayanarak anomali tespiti, pompaların arızalanma olasılığını önceden tahmin eden kestirimci bakım modelleri, ve su kalite parametrelerinden genel bir kalite skoru tahmini yapılabilir. Böylece arızalar oluşmadan müdahale edilerek hem çevresel hem de operasyonel riskler azaltılabilir, tesis verimliliği ve sürdürülebilirliği artırılabilir.

## Veri Seti

Kullanılan veri seti, gerçek bir arıtma tesisinden alınan sensör ölçümlerini içerir (Kaggle - Pump Sensor Data).

- **İçerik:** 52 farklı sensörden (RPM, titreşim, sıcaklık vb.) alınan zaman damgalı veriler.
- **Target:** `machine_status` (NORMAL, BROKEN, RECOVERING).

## Kurulum

Projeyi yerel ortamınızda çalıştırmak için:

```bash
# Repoyu klonlayın
git clone https://github.com/oznurkarahasan/wastewater-predictive-maintenance.git

# Sanal ortam oluşturun (Opsiyonel ama önerilir)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Gereksinimleri yükleyin
pip install -r requirements.txt
```

## Not:

Bazı yardımcı kod parçaları ve dokümantasyon süreçlerinde LLM tabanlı araçlardan yararlanılmıştır. Projenin ana geliştirme ve mimari kararları tamamen proje sahibine aittir.

## TO DO

- [x] GitHub Reposunun ve Klasör Yapısının Oluşturulması
- [x] GitHub Actions (CI/CD) Entegrasyonu
- [x] Veri setinin incelenmesi
- [x] Veri temizlenmesi ve görselleştirilmesi.
- [x] Temizlenmiş ham veri ile basit bir "Random Forest" modeli kurulması
- [x] Referans metriklerin (Recall, F1-Score) belirlenmesi.
- [x] Temel modelin kurulması ve iyileştirme çalışmaları (Feature Engineering).
- [x] Hareketli ortalamalar (Rolling Stats) ve gecikmeli değişkenler (Lags) üretme.
- [ ] Zaman serisi özelliklerini modele kazandırma.
- [ ] Random Forest yerine LightGBM veya XGBoost (Gradient Boosting) modellerinin kurulması.
- [ ] Modelin API servisine dönüştürülmesi ve Dashboard tasarımı.
- [ ] Sonuçların raporlanması ve sunum hazırlığı.

