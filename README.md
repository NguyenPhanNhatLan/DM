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

## Cấu trúc thư mục
```
project-root/
│
├── data/
│ ├── raw/ # Dữ liệu gốc
│ └── processed/ # Dữ liệu sau tiền xử lý
│
├── notebooks/ # Notebook phân tích & thử nghiệm
│
├── src/ # Code tiền xử lý & huấn luyện mô hình
│
├── dashboard/ # Ứng dụng Streamlit
│ ├── app.py
│ ├── pages/
│ └── services/
│
├── requirements.txt # Danh sách thư viện
├── README.md # Tài liệu hướng dẫn
└── .gitignore

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

