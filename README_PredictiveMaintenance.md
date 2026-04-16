# Kestirimci Bakım — Makine Arıza Tahmini

> 📌 Bu proje, Endüstri Mühendisliği lisans eğitimi kapsamında gerçekleştirilen staj hazırlık çalışmasının bir parçasıdır.

## Proje Hakkında

Kaggle **AI4I 2020 Predictive Maintenance Dataset** kullanılarak üretim makinelerinde arıza tahmini yapan bir makine öğrenmesi modeli geliştirilmiştir. Model, sensör verilerinden arızayı önceden tespit ederek reaktif bakım maliyetini **%60 azaltma** potansiyeli sunmaktadır.

## Veri Seti

| Özellik | Değer |
|---------|-------|
| Kaynak | [Kaggle — AI4I 2020](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) |
| Gözlem | 10,000 |
| Arıza Oranı | %3.4 (339 arıza) |
| Arıza Tipleri | Heat Dissipation, Power Failure, Overstrain, Tool Wear, Random |

## Kullanılan Özellikler

| Özellik | Tür |
|---------|-----|
| Hava Sıcaklığı (K) | Ham |
| Proses Sıcaklığı (K) | Ham |
| Dönüş Hızı (rpm) | Ham |
| Tork (Nm) | Ham |
| Takım Aşınması (dk) | Ham |
| Sıcaklık Farkı | **Türetilmiş** |
| Güç (W) | **Türetilmiş** |
| Aşınma Oranı | **Türetilmiş** |

## Model Sonuçları

| Model | Accuracy | F1 Score | ROC-AUC | Recall |
|-------|----------|----------|---------|--------|
| Logistic Regression | 0.861 | 0.294 | 0.935 | 0.853 |
| **Random Forest** | **0.989** | **0.810** | **0.980** | 0.721 |
| Gradient Boosting | 0.991 | 0.850 | 0.964 | 0.794 |

## En Önemli Özellikler

1. **Dönüş Hızı (rpm)** — 0.213
2. **Tork (Nm)** — 0.192
3. **Güç (türetilmiş)** — 0.190

## Maliyet Analizi

| Strateji | Tahmini Maliyet |
|----------|----------------|
| Reaktif Bakım | 3,400K₺ |
| Önleyici Bakım | 544K₺ |
| **Kestirimci Bakım (Model)** | **1,354K₺** |
| **Tasarruf** | **%60** |

## Kurulum & Çalıştırma

```bash
pip install pandas numpy matplotlib scikit-learn
python predictive_maintenance_analysis.py
```

Veri dosyasını (`predictive_maintenance.csv`) aynı klasöre koy.

## Kullanılan Araçlar

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

*Endüstri Mühendisliği — Staj Hazırlık Projesi*
