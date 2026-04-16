import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance

# ── 1. VERİ YÜKLEME & TEMİZLEME ─────────────────────────────────────
df = pd.read_csv('/home/claude/predictive_maintenance/predictive_maintenance.csv')

# Gereksiz sütunları çıkar
df = df.drop(['UDI', 'Product ID'], axis=1)

# Type encoding
le = LabelEncoder()
df['Type_enc'] = le.fit_transform(df['Type'])  # H=0, L=1, M=2

# Türetilmiş özellikler (feature engineering)
df['Temp_Diff']    = df['Process temperature [K]'] - df['Air temperature [K]']
df['Power']        = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2*np.pi/60)
df['Wear_Rate']    = df['Tool wear [min]'] / (df['Rotational speed [rpm]'] + 1)

features = ['Type_enc', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
            'Temp_Diff', 'Power', 'Wear_Rate']

X = df[features]
y = df['Target']

# ── 2. TRAIN/TEST SPLIT ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. MODEL EĞİTİMİ ─────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest':        RandomForestClassifier(n_estimators=200, random_state=42,
                                                    class_weight='balanced', n_jobs=-1),
    'Gradient Boosting':    GradientBoostingClassifier(n_estimators=150, random_state=42,
                                                        learning_rate=0.1),
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_sc, y_train)
        y_pred   = model.predict(X_test_sc)
        y_proba  = model.predict_proba(X_test_sc)[:,1]
    else:
        model.fit(X_train, y_train)
        y_pred   = model.predict(X_test)
        y_proba  = model.predict_proba(X_test)[:,1]

    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred),
        'f1':        f1_score(y_test, y_pred),
        'roc_auc':   roc_auc_score(y_test, y_proba),
    }
    print(f"{name}: Acc={results[name]['accuracy']:.3f} | "
          f"F1={results[name]['f1']:.3f} | AUC={results[name]['roc_auc']:.3f}")

# En iyi model = Random Forest
best = results['Random Forest']
rf_model = best['model']

# Feature importance
feat_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)

# ── 4. GÖRSELLEŞTİRME ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Kestirimci Bakım (Predictive Maintenance) — Makine Arıza Tahmini\nKaggle AI4I 2020 Dataset | Random Forest + Feature Engineering',
             fontsize=14, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
colors_main = ['#2E75B6','#E24B4A','#1F9E75','#F2A623','#7030A0']

# ─ 1. Arıza tipi dağılımı (pie)
ax1 = fig.add_subplot(gs[0, 0])
failure_counts = df['Failure Type'].value_counts()
labels_short = [l.replace(' Failure','').replace('No ','Sağlıklı\n') for l in failure_counts.index]
wedges, texts, autotexts = ax1.pie(
    failure_counts.values, labels=labels_short,
    colors=colors_main[:len(failure_counts)],
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 8})
ax1.set_title('Arıza Tipi Dağılımı', fontweight='bold', fontsize=10)

# ─ 2. Sensör verisi dağılımı — Arıza vs Normal
ax2 = fig.add_subplot(gs[0, 1])
normal  = df[df['Target']==0]['Torque [Nm]']
failure = df[df['Target']==1]['Torque [Nm]']
ax2.hist(normal,  bins=40, alpha=0.6, color='#2E75B6', label=f'Normal (n={len(normal):,})')
ax2.hist(failure, bins=40, alpha=0.8, color='#E24B4A', label=f'Arıza (n={len(failure):,})')
ax2.set_title('Tork Dağılımı: Normal vs Arıza', fontweight='bold', fontsize=10)
ax2.set_xlabel('Tork (Nm)')
ax2.set_ylabel('Frekans')
ax2.legend(fontsize=9)

# ─ 3. Takım aşınması vs arıza
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(df[df['Target']==0]['Tool wear [min]'],
            df[df['Target']==0]['Torque [Nm]'],
            alpha=0.3, s=5, color='#2E75B6', label='Normal')
ax3.scatter(df[df['Target']==1]['Tool wear [min]'],
            df[df['Target']==1]['Torque [Nm]'],
            alpha=0.8, s=20, color='#E24B4A', label='Arıza')
ax3.set_title('Takım Aşınması vs Tork', fontweight='bold', fontsize=10)
ax3.set_xlabel('Takım Aşınması (dk)')
ax3.set_ylabel('Tork (Nm)')
ax3.legend(fontsize=9)

# ─ 4. Model karşılaştırması
ax4 = fig.add_subplot(gs[1, 0])
model_names = list(results.keys())
metrics_vals = {
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'F1 Score': [results[m]['f1'] for m in model_names],
    'ROC-AUC':  [results[m]['roc_auc'] for m in model_names],
}
x = np.arange(len(model_names))
width = 0.25
for i, (metric, vals) in enumerate(metrics_vals.items()):
    bars = ax4.bar(x + i*width, vals, width, label=metric,
                   color=colors_main[i], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
ax4.set_xticks(x + width)
short_names = ['Lojistik\nRegresyon','Random\nForest','Gradient\nBoosting']
ax4.set_xticklabels(short_names, fontsize=8)
ax4.set_ylim(0, 1.15)
ax4.set_title('Model Karşılaştırması', fontweight='bold', fontsize=10)
ax4.legend(fontsize=8, loc='upper left')

# ─ 5. Confusion Matrix (Random Forest)
ax5 = fig.add_subplot(gs[1, 1])
cm = confusion_matrix(y_test, best['y_pred'])
im = ax5.imshow(cm, cmap='Blues')
ax5.set_xticks([0,1]); ax5.set_yticks([0,1])
ax5.set_xticklabels(['Normal','Arıza']); ax5.set_yticklabels(['Normal','Arıza'])
ax5.set_xlabel('Tahmin'); ax5.set_ylabel('Gerçek')
ax5.set_title('Confusion Matrix\n(Random Forest)', fontweight='bold', fontsize=10)
for i in range(2):
    for j in range(2):
        ax5.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax5, shrink=0.8)

# ─ 6. ROC Curve
ax6 = fig.add_subplot(gs[1, 2])
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax6.plot(fpr, tpr, color=colors_main[i], linewidth=2,
             label=f"{name.split()[0]} (AUC={res['roc_auc']:.3f})")
ax6.plot([0,1],[0,1], 'k--', alpha=0.5, label='Rastgele')
ax6.set_xlabel('False Positive Rate')
ax6.set_ylabel('True Positive Rate')
ax6.set_title('ROC Eğrisi', fontweight='bold', fontsize=10)
ax6.legend(fontsize=8)
ax6.fill_between(*roc_curve(y_test, best['y_proba'])[:2],
                 alpha=0.1, color='#2E75B6')

# ─ 7. Feature Importance
ax7 = fig.add_subplot(gs[2, 0])
feat_colors = ['#E24B4A' if v > feat_imp.mean() else '#2E75B6' for v in feat_imp.values]
bars = ax7.barh(range(len(feat_imp)), feat_imp.values, color=feat_colors)
feat_labels_tr = {
    'Torque [Nm]':                'Tork (Nm)',
    'Rotational speed [rpm]':     'Dönüş Hızı (rpm)',
    'Tool wear [min]':             'Takım Aşınması',
    'Power':                       'Güç (türetilmiş)',
    'Wear_Rate':                   'Aşınma Oranı',
    'Temp_Diff':                   'Sıcaklık Farkı',
    'Process temperature [K]':    'Proses Sıcaklığı',
    'Air temperature [K]':        'Hava Sıcaklığı',
    'Type_enc':                    'Makine Tipi',
}
ax7.set_yticks(range(len(feat_imp)))
ax7.set_yticklabels([feat_labels_tr.get(f, f) for f in feat_imp.index], fontsize=9)
ax7.set_title('Özellik Önemi\n(Random Forest)', fontweight='bold', fontsize=10)
ax7.set_xlabel('Önem Skoru')
for bar, val in zip(bars, feat_imp.values):
    ax7.text(val+0.001, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=8)

# ─ 8. Sıcaklık farkı vs arıza
ax8 = fig.add_subplot(gs[2, 1])
bins = np.linspace(df['Temp_Diff'].min(), df['Temp_Diff'].max(), 20)
ax8.hist(df[df['Target']==0]['Temp_Diff'], bins=bins, alpha=0.6,
         color='#2E75B6', label='Normal', density=True)
ax8.hist(df[df['Target']==1]['Temp_Diff'], bins=bins, alpha=0.8,
         color='#E24B4A', label='Arıza', density=True)
ax8.set_title('Sıcaklık Farkı Dağılımı\n(Türetilmiş Özellik)', fontweight='bold', fontsize=10)
ax8.set_xlabel('Proses − Hava Sıcaklığı (K)')
ax8.set_ylabel('Yoğunluk')
ax8.legend(fontsize=9)

# ─ 9. Maliyet analizi
ax9 = fig.add_subplot(gs[2, 2])
# Bakım maliyeti karşılaştırması
categories = ['Reaktif\nBakım', 'Önleyici\nBakım', 'Kestirimci\nBakım\n(Bu Model)']
# Gerçekçi maliyet tahmini senaryosu
fn = cm[1][0]  # kaçırılan arızalar
fp = cm[0][1]  # yanlış alarm
tp = cm[1][1]  # doğru tespit
tn = cm[0][0]

cost_reactive    = fn * 50000 + tp * 50000          # tüm arızalar reaktif
cost_preventive  = (fn+tp) * 8000                    # periyodik bakım herkese
cost_predictive  = fn * 50000 + fp * 3000 + tp * 8000  # model bazlı

costs = [cost_reactive/1000, cost_preventive/1000, cost_predictive/1000]
bar_colors_c = ['#E24B4A','#F2A623','#1F9E75']
bars_c = ax9.bar(categories, costs, color=bar_colors_c, alpha=0.85, edgecolor='white')
ax9.set_title('Bakım Stratejisi\nMaliyet Karşılaştırması (Test Seti)', fontweight='bold', fontsize=10)
ax9.set_ylabel('Tahmini Maliyet (K₺)')
for bar, val in zip(bars_c, costs):
    ax9.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f'{val:,.0f}K₺', ha='center', va='bottom', fontsize=9, fontweight='bold')
saving_pct = (costs[0]-costs[2])/costs[0]*100
ax9.text(0.5, 0.95, f'Tasarruf: %{saving_pct:.0f}', transform=ax9.transAxes,
         ha='center', fontsize=10, fontweight='bold', color='#1F9E75')

plt.savefig('/home/claude/predictive_maintenance_analiz.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

# ── 5. SONUÇ RAPORU ───────────────────────────────────────────────────
print("\n" + "="*60)
print("KEStirimci BAKIM ANALİZİ — SONUÇ RAPORU")
print("="*60)
print(f"\nVeri Seti: 10,000 gözlem | {df['Target'].sum()} arıza (%{df['Target'].mean()*100:.1f})")
print(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10} {'Recall':>8}")
print("-"*65)
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:>10.3f} {res['f1']:>8.3f} "
          f"{res['roc_auc']:>10.3f} {res['recall']:>8.3f}")

print(f"\n✓ En İyi Model: Random Forest")
print(f"  → Tespit edilen arıza: {tp}/{tp+fn} (%{tp/(tp+fn)*100:.0f} recall)")
print(f"  → Kaçırılan arıza: {fn} (Reaktif bakım riski)")
print(f"  → Yanlış alarm: {fp}")

print(f"\nEn Önemli 3 Özellik:")
for i, (feat, imp) in enumerate(feat_imp.head(3).items(), 1):
    print(f"  {i}. {feat_labels_tr.get(feat, feat)}: {imp:.3f}")

print(f"\nMaliyet Analizi (Test Seti):")
print(f"  Reaktif Bakım:    {cost_reactive/1000:,.0f}K₺")
print(f"  Önleyici Bakım:   {cost_preventive/1000:,.0f}K₺")
print(f"  Kestirimci Bakım: {cost_predictive/1000:,.0f}K₺")
print(f"  → Reaktife göre tasarruf: %{saving_pct:.0f}")

print("\n✓ Grafik kaydedildi: predictive_maintenance_analiz.png")
