%% KALP HASTALIĞI RİSK TAHMİNİ VE ANALİZ PROJESİ
% Bu script, veri setini yükler, temizler, görselleştirir ve SVM modelini eğitir.

clc; clear; close all;
disp('--- PROJE BAŞLATILIYOR ---');

%% 1. VERİ YÜKLEME VE HAZIRLIK
disp('1. Veri Seti Yükleniyor...');

% '?' ve 'NaN' değerlerini eksik veri olarak tanıtarak yükleme
opts = detectImportOptions('heart.csv', 'TreatAsMissing', {'?', 'NaN'});
opts = setvartype(opts, {'ca', 'thal'}, 'double'); % Hata veren sütunları düzelt
T = readtable('heart.csv', opts);

% Gereksiz sütunları çıkarma (id, dataset)
if ismember('id', T.Properties.VariableNames)
    T.id = [];
end
if ismember('dataset', T.Properties.VariableNames)
    T.dataset = [];
end

%% 2. EKSİK VERİ TEMİZLEME VE DÖNÜŞÜM
disp('2. Veri Temizleniyor...');

% Sadece sayısal sütunları seç ve medyan ile doldur
numeric_cols = {'age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'thal'};
for i = 1:length(numeric_cols)
    col_name = numeric_cols{i};
    if isnumeric(T.(col_name))
        T.(col_name) = fillmissing(T.(col_name), 'movmedian', 5);
    end
end

% Hedef Değişkeni Oluşturma (0: Yok, 1-4: Var -> İkili Sınıf)
% num sütunu hedef değişkendir
Y = categorical(T.num > 0, [false, true], {'No', 'Yes'}); 

disp('   -> Eksik veriler dolduruldu.');
disp('   -> Hedef değişken ikili sınıfa dönüştürüldü.');

%% 3. NORMALİZASYON (ÖLÇEKLEME)
disp('3. Özellikler Normalize Ediliyor (Z-Score)...');

% Eğitimde kullanılacak 5 temel özellik
features_to_use = {'age', 'trestbps', 'chol', 'thalch', 'oldpeak'};
X_raw = T(:, features_to_use);

% Normalizasyon parametrelerini hesapla (App Designer için gerekli!)
M_train = mean(table2array(X_raw), 1);
S_train = std(table2array(X_raw), 0, 1);

% Veriyi normalize et
X_norm = normalize(table2array(X_raw));

fprintf('   -> Ortalama (Mean): %s\n', num2str(M_train));
fprintf('   -> Std Sapma (Std): %s\n', num2str(S_train));

%% 4. VERİ GÖRSELLEŞTİRME (EDA)
disp('4. Grafikler Oluşturuluyor...');

% A. Sınıf Dağılımı (Bar Chart)
figure('Name', 'Sınıf Dağılımı');
histogram(Y);
title('Hasta ve Sağlıklı Birey Dağılımı');
xlabel('Durum'); ylabel('Kişi Sayısı');
grid on;

% B. Korelasyon Isı Haritası (Heatmap)
figure('Name', 'Korelasyon Haritası');
T_corr = T(:, features_to_use);
T_corr.Target = T.num; % Hedefi de ekle
R = corr(table2array(T_corr), 'rows', 'complete');
heatmap(T_corr.Properties.VariableNames, T_corr.Properties.VariableNames, R);
title('Özellikler Arası Korelasyon');

% C. Yaş ve Hastalık İlişkisi (Box Plot)
figure('Name', 'Yaş Analizi');
boxplot(T.age, Y);
title('Hastalık Durumuna Göre Yaş Dağılımı');
ylabel('Yaş');

%% 5. MODEL EĞİTİMİ (SVM)
disp('5. SVM Modeli Eğitiliyor...');

% Veriyi Eğitim (%70) ve Test (%30) olarak ayır
cv = cvpartition(Y, 'Holdout', 0.3);
XTrain = X_norm(cv.training, :);
YTrain = Y(cv.training, :);
XTest = X_norm(cv.test, :);
YTest = Y(cv.test, :);

% SVM Modelini Eğit (Gaussian Kernel)
mdl_svm_final = fitcsvm(XTrain, YTrain, 'KernelFunction', 'gaussian', ...
    'Standardize', false, 'BoxConstraint', 1);

% Test Verisi ile Tahmin Yap
[YPred, scores] = predict(mdl_svm_final, XTest);

% Doğruluk Hesapla
accuracy = sum(YPred == YTest) / length(YTest);
fprintf('   -> MODEL DOĞRULUĞU: %.2f%%\n', accuracy * 100);

%% 6. MODEL PERFORMANS GÖRSELLEŞTİRME
% A. Karmaşıklık Matrisi
figure('Name', 'Model Performansı');
confusionchart(YTest, YPred);
title(['SVM Başarısı (Doğruluk: %' num2str(accuracy*100, '%.1f') ')']);

% B. ROC Eğrisi
figure('Name', 'ROC Eğrisi');
roc_obj = rocmetrics(YTest, scores(:,2), 'Yes');
plot(roc_obj);
title(['ROC Eğrisi - AUC: ' num2str(roc_obj.AUC, '%.3f')]);

%% 7. MODELİ KAYDETME
disp('7. Model Kaydediliyor...');

% Modeli tek bir değişkene ata ve kaydet
mdl_final = mdl_svm_final;
save('kalp_tahmin_modeli_final.mat', 'mdl_final');

disp('--- İŞLEM TAMAMLANDI. MODEL APP DESIGNER İÇİN HAZIR ---');