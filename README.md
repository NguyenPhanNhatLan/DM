# Phân tích hành vi khách hàng và tối ưu hóa hiệu quả chiến dịch telemarketing cho sản phẩm tiền gửi có kỳ hạn tại ngân hàng

## Tổng quan đồ án
- Bộ dữ liệu: Bank Marketing Dataset (Ngân hàng Bồ Đào Nha)
- Bài toán:
  - Tối đa hóa số lượng khách hàng đăng ký tiền gửi có kỳ hạn 
  - Giảm chi phí và nguồn lực cho các cuộc gọi không hiệu quả
- Mục tiêu phân tích:
  - Dự đoán khách hàng đồng ý đăng ký (y = yes)
  - Xác định các yếu tố ảnh hưởng mạnh đến khả năng chuyển đổi
  - Rút ra tri thức vận hành 
- Mô hình sử dụng:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Trực quan hóa:
  - Dashboard tương tác bằng Streamlit

## Theo dõi huấn luyện mô hình & kết quả trực quan

### MLflow
MLflow được sử dụng để theo dõi toàn bộ quá trình huấn luyện mô hình, bao gồm:
- So sánh các bộ tham số
- Đánh giá hiệu quả mô hình qua nhiều chỉ số (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Lưu trữ và quản lý mô hình tốt nhất
[https://mlflow.thonph.site/]
![Uploading image.png…]()


### Trực quan hóa tri thức
Dashboard dưới đây tổng hợp các tri thức rút ra từ quá trình khai phá dữ liệu (data mining), bao gồm phân tích khách hàng, chiến dịch, và cả hai.
<img width="2856" height="1369" alt="Screenshot 2025-12-25 211801" src="https://github.com/user-attachments/assets/c86207b2-bd75-4581-ae36-aad16ad8da19" />

## Cấu trúc thư mục
```
project-root/
│
├── data/
│ └── processed/ # Dữ liệu sau xử lý cho data_mining, models và actions
│
├── notebooks/ # Notebook phân tích & thử nghiệm
│ ├── eda # Tiền xử lý, khám phá dữ liệu
│ ├── data_mining # Khám phá tri thức từ campaign, customer, tổng hợp
│ ├── models # Huấn luyện mô hình
| └── actions # Khuyến nghị hành động từ tri thức và dự báo từ model
│
├── dashboard/ # Ứng dụng Streamlit
│ ├── app.py
│ ├── pages/
│ └── services/
│
├── requirements.txt # Danh sách thư viện
├── README.md # Tài liệu hướng dẫn
└── .gitignore
```
## Môi trường thực thi
- Python: **3.9 trở lên**
- Hệ điều hành: Windows / Linux / macOS
- Thư viện chính:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - plotly
  - streamlit

Toàn bộ thư viện được liệt kê trong `requirements.txt`.

## Cài đặt

Clone repo và cài đặt môi trường:

```bash
git clone <link-repo>
cd project-root
pip install -r requirements.txt

