# Yeni keşifsel analizler, 2026-11-05

> Version 1.2.6 note: this is the dated round-1 execution record. Its four follow-ups are now documented in Supplementary Sections S3, S4 and S9 and docs/TECHNICAL_METHODS.md. Executed analysis code and numerical outputs are preserved.

Bu sonuçlar makaleye, eke veya okuma kopyasına eklenmedi. Tüm tasarımlar
kendi çalıştırmalarından önce tarihli plan dosyalarına yazıldı. Olumsuz ve
olumlu sonuçların hepsi aşağıdadır. Pozitif farklar EEG lehinedir.

ΔLL ve ΔBrier baseline eksi EEG; ΔAUROC EEG eksi baseline'dır. Metrikler
önce her tekrarın katılımcılarında, sonra tekrarlar üzerinden ortalandı.
Aralıklar, katılımcı başına tekrar-ortalama kayıp farklarından 10000
sınıf-tabakalı katılımcı bootstrap örnekleminin %2.5 ve %97.5 yüzdelikleridir
(PCG64, seed=20261105, doğrusal yüzdelik). Model, seçim ve tahminler sabittir;
bu aralıklar tüm öğrenme prosedürünün belirsizliğini kapsamaz.

## F1. Tanı hedefinde entropy/Hjorth

84 katılımcı: 54 yes, 30 no. Donmuş dokuz entropy/Hjorth ve yedi baseline
özelliği; her iki iç içe düzeyde DIAGNOSED_FOLD_MANIFEST. DQ, iki modele de
cihaz, temiz pencere oranı ve interpolasyon sayısı eklenmesidir.

| Karşılaştırma | Baseline LL | EEG LL | ΔLL | ΔAUROC | ΔBrier | Pozitif tekrar | Koşullu %95 LL aralığı |
|---|---:|---:|---:|---:|---:|---:|---|
| diagnosis_entropy | 0.5971 | 0.5877 | +0.0094 | +0.0022 | +0.0018 | 7/10 | [-0.0592, +0.0778] |
| diagnosis_entropy_DQ | 0.5822 | 0.5712 | +0.0110 | +0.0022 | +0.0041 | 7/10 | [-0.0577, +0.0744] |

Plan: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\diagnosis\PLAN_2026-11-05.md
Kod: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\diagnosis\run_diagnosis.py
Çıktılar: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\diagnosis\outputs

Ayarsız diagnosis baseline tekrarlardaki kayıtlı kayıpları en fazla
2.81e-13 hata ile yeniden üretti.
İki kazanç da küçüktür ve koşullu aralıkları sıfırı içerir. Hedef, klinik
yeniden değerlendirme değildir; uyuşmazlık içeren kayıtların diagnosis
alanına dayanır. DQ, EEG'den türediği için grup bilgisini de çıkarabilir.

## F2. EEGNet çapraz uydurma duyarlılığı

250 kayıtlı iç encoder yeniden eğitilmeden kullanıldı. Embedding uzaylarını
birleştirmek yerine encoder'ın kendi son eğitim başının logiti kullanıldı.
Her baş-eğitim skoru, o katılımcıyı encoder eğitiminde görmemiş bir modelden
geldi; dış-test katılımcıları hiçbir encoder veya baş eğitiminde yoktu.

split_ensemble: her encoder'ın dışarıda bıraktığı 15-16 katılımcıda iki ayrı
lojistik baş kuruldu, yalnız bu altkümede en fazla beş katlı CV ile C seçildi.
Beş başın dış-test olasılıkları ortalandı. Baseline aynı küçük altkümeleri ve
aynı toplulaştırmayı kullanır. Böylece başın C doğrulama etiketleri de ilgili
encoder eğitimine hiç girmedi.

pooled_oof_fixed_C1: dış-eğitimin 76-77 katılımcısının kendi iç-valid logitleri
birleştirildi; C önceden 1'e sabitlendi, ayarlanmadı. Bu, OOF özellikler için
yeni iç-C aramasındaki dolaylı etiket bağımlılığını önler. Baş, testte her
encoder'ın logitine ayrı uygulandı; beş olasılık ortalandı. Eşleşmiş baseline
da C=1 kullanır. Skorların diğer eğitim katılımcılarının etiketlerine
bağımlılığı olağan çapraz uydurma yapısında kalır; dış test bağımsızdır.

| Karşılaştırma | Baseline LL | EEG LL | ΔLL | ΔAUROC | ΔBrier | Pozitif tekrar | Koşullu %95 LL aralığı |
|---|---:|---:|---:|---:|---:|---:|---|
| split_ensemble | 0.6476 | 0.6457 | +0.0019 | +0.0047 | +0.0008 | 8/10 | [-0.0092, +0.0130] |
| split_ensemble_vs_recorded_full_baseline | 0.6244 | 0.6457 | -0.0213 | -0.0462 | -0.0116 | 1/10 | [-0.0650, +0.0275] |
| pooled_oof_fixed_C1 | 0.6146 | 0.6093 | +0.0053 | +0.0097 | +0.0026 | 8/10 | [-0.0151, +0.0250] |
| pooled_oof_vs_recorded_full_baseline | 0.6244 | 0.6093 | +0.0151 | +0.0200 | +0.0069 | 8/10 | [-0.0199, +0.0469] |
| original_embedding_reference | 0.6244 | 0.7250 | -0.1006 | -0.0496 | -0.0287 | 0/10 | [-0.2404, +0.0252] |

Plan: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\eegnet_crossfit\PLAN_2026-11-05.md
Encoder çıkarımı: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\eegnet_crossfit\infer_saved_encoders.py
Baş değerlendirmesi: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\eegnet_crossfit\evaluate_heads.py
Çıktılar: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\eegnet_crossfit\outputs

_vs_recorded_full_baseline satırları, yeni EEG modelini makaledeki tam-eğitim
baseline'ıyla karşılaştıran ek tanılardır; bunlar eşleşmiş baş tasarımları
değildir. original_embedding_reference yalnızca kayıtlı 128-boyutlu embedding
sonucudur. Bu referans satırının aralığı raporun ortak bootstrap yöntemiyle
yeniden hesaplandı; makaledeki kayıtlı aralık değiştirilmedi.

Baş-eğitim skoru/encoder-eğitim katılımcı kesişimi 0; yeni encoder eğitimi 0.
Kaydedilmiş embedding ile CPU yeniden çıkarımı arasındaki en büyük fark
7.31e-07 idi. CPU torch 2.5.1 ve kayıtlı ağırlıklar kullanıldı.

Sınır: 128 embedding yerine tek logit, farklı baş-eğitim büyüklükleri ve
ensemble yapısı kullanılıyor. Dolayısıyla eski kayıp artışının azalmasını
yalnız başın aynı katılımcıların embedding'leriyle kurulmasına bağlayamayız.
Küçük pozitif eşleşmiş farkların bütün koşullu aralıkları sıfırı içerir.
Tam baseline'a karşı split ensemble hâlâ olumsuzdur. Bu karşılaştırmalar
ne bir EEGNet üstünlüğü ne de mimariler arasında en iyi tasarımı belirler.

## F3. Phase-lag için aynı 20 faz kontrolü

Kayıtlı 20 veri setinin sinyal dizileri saklanmadığı için özgün float64 kabul
edilmiş pencereler ham dosyalardan yeniden çıkarıldı; aynı kayıtlı tohumlarla
kontroller yeniden oluşturuldu. Saklanan entropy/Hjorth kontrol özellikleri
20 veri setinin hepsinde tam eşleşti. Yeni dWPLI özellikleri 273 değer;
eğitim içinde band başına üç PCA bileşeni ve aynı iç içe lojistik prosedür
kullanıldı. Kontroller arasında seçim yapılmadı.

| Karşılaştırma | Baseline LL | EEG LL | ΔLL | ΔAUROC | ΔBrier | Pozitif tekrar | Koşullu %95 LL aralığı |
|---|---:|---:|---:|---:|---:|---:|---|
| observed | 0.6244 | 0.6117 | +0.0127 | +0.0031 | +0.0035 | 7/10 | [-0.0204, +0.0501] |
| phase_control_00 | 0.6244 | 0.6348 | -0.0104 | -0.0175 | -0.0036 | 4/10 | [-0.0344, +0.0153] |
| phase_control_01 | 0.6244 | 0.6314 | -0.0071 | -0.0199 | -0.0033 | 2/10 | [-0.0298, +0.0155] |
| phase_control_02 | 0.6244 | 0.6165 | +0.0079 | +0.0068 | +0.0031 | 7/10 | [-0.0213, +0.0390] |
| phase_control_03 | 0.6244 | 0.6430 | -0.0186 | -0.0330 | -0.0088 | 1/10 | [-0.0443, +0.0081] |
| phase_control_04 | 0.6244 | 0.6341 | -0.0097 | -0.0117 | -0.0042 | 3/10 | [-0.0347, +0.0169] |
| phase_control_05 | 0.6244 | 0.6490 | -0.0246 | -0.0445 | -0.0111 | 1/10 | [-0.0474, +0.0011] |
| phase_control_06 | 0.6244 | 0.6362 | -0.0118 | -0.0320 | -0.0073 | 3/10 | [-0.0355, +0.0135] |
| phase_control_07 | 0.6244 | 0.6201 | +0.0043 | +0.0034 | +0.0017 | 6/10 | [-0.0252, +0.0351] |
| phase_control_08 | 0.6244 | 0.6558 | -0.0314 | -0.0431 | -0.0151 | 0/10 | [-0.0517, -0.0094] |
| phase_control_09 | 0.6244 | 0.6387 | -0.0144 | -0.0102 | -0.0050 | 2/10 | [-0.0432, +0.0143] |
| phase_control_10 | 0.6244 | 0.6512 | -0.0268 | -0.0439 | -0.0126 | 0/10 | [-0.0522, -0.0007] |
| phase_control_11 | 0.6244 | 0.6275 | -0.0031 | -0.0142 | -0.0026 | 4/10 | [-0.0327, +0.0281] |
| phase_control_12 | 0.6244 | 0.6546 | -0.0302 | -0.0534 | -0.0159 | 2/10 | [-0.0536, -0.0043] |
| phase_control_13 | 0.6244 | 0.6027 | +0.0217 | +0.0362 | +0.0109 | 8/10 | [-0.0195, +0.0639] |
| phase_control_14 | 0.6244 | 0.6463 | -0.0220 | -0.0206 | -0.0076 | 1/10 | [-0.0472, +0.0044] |
| phase_control_15 | 0.6244 | 0.6434 | -0.0190 | -0.0311 | -0.0090 | 2/10 | [-0.0450, +0.0081] |
| phase_control_16 | 0.6244 | 0.6289 | -0.0045 | -0.0243 | -0.0030 | 3/10 | [-0.0315, +0.0256] |
| phase_control_17 | 0.6244 | 0.6252 | -0.0009 | -0.0024 | +0.0006 | 5/10 | [-0.0417, +0.0386] |
| phase_control_18 | 0.6244 | 0.6433 | -0.0189 | -0.0277 | -0.0072 | 2/10 | [-0.0427, +0.0070] |
| phase_control_19 | 0.6244 | 0.6334 | -0.0091 | -0.0239 | -0.0055 | 2/10 | [-0.0343, +0.0174] |

Gözlenen ΔLL: +0.012678.
Kontrol ortalaması: -0.011429; kontrol aralığı:
[-0.031445, +0.021695].
Gözlenen eksi kontrol ortalaması: +0.024107.
Pozitif kontrol ortalaması 3/20; gözlenenden yüksek 1/20.

Plan: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\phase_lag\PLAN_2026-11-05.md
Özellik çıkarımı: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\phase_lag\extract_phase_lag.py
Değerlendirme: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\phase_lag\evaluate_phase_lag.py
Çıktılar: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\phase_lag\outputs
Her kontrolün tahmin, seçim ve tekrar kayıtları: F:\proje\eeg_gain\private_work\cmpb_r2_2026-11-05\repo\exploratory_r2_2026-11-05\phase_lag\evaluation

Gözlenen E3 yeniden değerlendirmesi kayıtlı EEG tahminlerini en fazla
6.02e-12, baseline tahminlerini en fazla
1.96e-11 hata ile üretti.
Özgün E3 ve yeniden oluşturulan kontrol E4 özelliklerindeki en büyük farklar
sırasıyla 0 ve 0; Fourier genlik bağıl hatası en çok
4.56e-16. Hızlandırılmış dWPLI çıkarımı donmuş fonksiyonla da doğrulandı.

Sınır: 20 kontrol aynı katılımcıların aynı kabul edilmiş pencerelerinden
türetilir; bağımsız kohortlar değildir. Kontroller 08, 10 ve 12'nin koşullu
aralıkları tamamen negatiftir; diğerlerinin aralıkları sıfırı içerir.
Bu betimsel karşılaştırma yeni bir anlamlılık testi değildir. Faz kontrolleri
çapraz-kanal fazı yanında pencere içi yapıyı da değiştirir; farkı tek bir
fizyolojik mekanizmaya atfetmez. Gözlenen kazancın koşullu aralığı da sıfırı
içerir. Bu yeni sonuçlar makaleye eklenmedi.
