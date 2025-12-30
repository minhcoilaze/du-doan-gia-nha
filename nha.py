import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#đọc data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print("1. Kích thước dữ liệu:", train_data.shape) #
print("2. Kiểu dữ liệu:\n", train_data.info()) #
print("3. Số lượng NaN trên mỗi cột:\n", train_data.isnull().sum()) #
print("4. Số lượng dòng trùng lặp:", train_data.duplicated().sum()) #
print("5. Thống kê mô tả:\n", train_data.describe())

#tiền xử lý dữ liệu
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

x = train_data.drop(columns = ['Id', 'SalePrice'])
y = train_data['SalePrice']
test_data = test_data.drop(columns = ['Id'])

x['MSSubClass'] = x['MSSubClass'].astype(str)
test_data['MSSubClass'] = test_data['MSSubClass'].astype(str)

cot_so = x.select_dtypes(include = ['int64', 'float64']).columns
cot_chu = x.select_dtypes(include = ['object']).columns

xu_ly_so = Pipeline(steps = [
    ('buoc_1', SimpleImputer(strategy = 'median')),
    ('buoc_2', StandardScaler())
])
xu_ly_chu = Pipeline(steps = [
    ('buoc_1', SimpleImputer(strategy = 'constant', fill_value = 'None')),
    ('buoc_2', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False))
])
xu_ly_ca = ColumnTransformer(transformers = [
    ('buoc_1', xu_ly_so, cot_so),
    ('buoc_2', xu_ly_chu, cot_chu)
])

#chia data
from sklearn.model_selection import train_test_split, GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, 
                                                    random_state = 42)

#chọn mô hình và train
from sklearn.ensemble import RandomForestRegressor
model = Pipeline(steps = [
    ('buoc_1', xu_ly_ca),
    ('buoc_2', RandomForestRegressor(random_state = 42))
])
chinh_sieu_ts = {
    'buoc_2__n_estimators' : [50, 70, 100, 150, 200],
    'buoc_2__max_depth' : [None, 3, 5, 7, 15, 20],
    'buoc_2__min_samples_split' : [2, 5]
}
luoi_tham_so = GridSearchCV(
    estimator = model,
    param_grid = chinh_sieu_ts,
    cv = 5,
    scoring = 'neg_mean_squared_error',
    n_jobs = -1
)
luoi_tham_so.fit(x_train, y_train)

#đánh giá mô hình
from sklearn.metrics import mean_squared_error, r2_score
print("Tham số tốt nhất:", luoi_tham_so.best_params_)

best_model = luoi_tham_so.best_estimator_
y_pred = best_model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE (Sai số trung bình): ${rmse:,.2f}")
print(f"R2 Score (Độ phù hợp): {r2:.4f}")

#vẽ 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Đường chéo đỏ
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('So sánh Giá thực tế vs Giá dự đoán')
plt.show()

#dự đoán trên tập test data
test_pred = best_model.predict(test_data)

