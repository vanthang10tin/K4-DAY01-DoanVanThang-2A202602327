# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy :** 2026-09-11

**Runtime Colab:** GPU (Tesla T4)

**Python / PyTorch / Ultralytics:** Python 3.10.12 / PyTorch 2.5.1+cu121 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  `{"class_id": 468, "class_name": "cab", "rank": 1, "score": 0.510915, "taxonomy_name": "ImageNet-1K"}`
- Record này mô tả toàn ảnh như thế nào?
  Record này gán một nhãn phân loại duy nhất (`cab` - xe taxi) đại diện cho toàn bộ nội dung của bức ảnh. Mô hình không định vị tọa độ hộp, không chỉ ra vị trí xuất hiện hay số lượng thực thể cụ thể, mà chỉ ước lượng phân phối xác suất tổng thể trên không gian 1.000 lớp của ImageNet-1K.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Danh sách này được định nghĩa bởi bộ dữ liệu dùng để huấn luyện mô hình của "ImageNet-1K
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  Cần giữ cả ba để đảm bảo tính toàn vẹn: ID dùng cho máy móc xử lý nhanh, tên lớp để con người có thể đọc hiểu, và tên taxonomy giúp xác định nguồn gốc chuẩn của từ vựng (tránh trùng lặp tên lớp giữa các bộ dữ liệu khác nhau)
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Trong bài toán phân loại đơn nhãn, khi bức ảnh chứa nhiều đối tượng khác nhau (như ảnh `traffic` gồm cả xe buýt lớn, xe con, xe tải, người đi bộ), guideline bắt buộc phải quy định quy tắc ưu tiên rõ ràng:
  1. Tiêu chí chủ thể nổi bật: Ưu tiên chọn đối tượng chiếm diện tích lớn nhất, nằm ở vị trí trung tâm, hoặc nằm ở tiền cảnh.
  2. Thứ tự ưu tiên: Xác định rõ cấp độ ưu tiên khi các nhóm đối tượng cùng xuất hiện.
  3. Quy tắc xử lý ngoại lệ: Nếu bức ảnh là đại cảnh phức tạp không có một chủ thể duy nhất chiếm ưu thế, guideline phải quy định chuyển sang bài toán đa nhãn (multi-label), chuyển sang phân loại cảnh, hoặc cho phép annotator đánh dấu ngoại lệ.
- Vì sao model score không phải ground truth?
  Model score (ở đây là softmax probability `0.510915`) chỉ là giá trị độ tin cậy thống kê. Nó phản ánh mức độ tự tin toán học của mô hình, không phải sự thật khách quan của thế giới thực.
  - Mô hình hoàn toàn có thể đưa ra score rất cao cho một dự đoán sai, như trong ảnh `traffic` chủ thể chiếm thị giác lớn nhất là các xe buýt công cộng, nhưng model lại dự đoán lớp hạng 1 là xe taxi.
  - Ground Truth là nhãn chuẩn xác thực tế do con người thẩm định, xác nhận độc lập và tuân thủ nghiêm ngặt theo guideline của bài toán. Tuyệt đối không được đồng nhất model score với chất lượng nhãn hoặc sao chép model score làm ground truth.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  `{"class_name": "person", "score": 0.912625, "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}`

- Diễn giải vị trí box bằng lời:
  Bounding box được định nghĩa bằng một hình chữ nhật bao quanh đối tượng, xác định bởi tọa độ pixel của góc trên cùng bên trái và góc dưới cùng bên phải (bbox_xyxy), cùng với chiều rộng và chiều cao của hộp
  - Với record `person`:
    - Góc trên-trái của box nằm tại `(x_min = 385.33, y_min = 69.24)`, tương ứng với vị trí đỉnh đầu của người đầu bếp ở nửa bên phải bức ảnh.
    - Góc dưới-phải của box nằm tại `(x_max = 498.92, y_max = 348.92)`, tương ứng với vị trí gót chân người đầu bếp tiếp xúc với sàn nhà.
    - Kích thước hộp: chiều rộng `bbox_width = 113.58 px`, chiều cao `bbox_height = 279.68 px`.
- So sánh số prediction ở hai threshold:
  - Ở ngưỡng `threshold = 0.35`: Mô hình `yolo11n.pt` phát hiện được 11 vật thể trong ảnh `kitchen` gồm: 2 `person` (scores: 0.91, 0.61), 2 `oven` (0.69, 0.63), 5 `bowl` (0.72, 0.70, 0.50, 0.46, 0.38), 2 `cup` (0.45, 0.38).
  - Ở ngưỡng `threshold = 0.60`: Mô hình chỉ giữ lại 6 vật thể có score >= 0.60 gồm: 2 `person` (0.91, 0.61), 2 `oven` (0.69, 0.63), 2 `bowl` (0.72, 0.70). Có 5 vật thể có độ tin cậy thấp hơn (gồm 3 chiếc `bowl` nhỏ và 2 chiếc `cup`) đã bị loại bỏ hoàn toàn.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Độ bao phủ : Nâng threshold làm giảm độ bao phủ của mô hình. Hạ threshold giúp tăng độ bao phủ (bắt được nhiều đối tượng tiềm năng hơn).
  - Khối lượng reviewer cần xem:
    - Khi hạ threshold: Reviewer nhận được nhiều hộp đề xuất hơn, nhưng phải tốn nhiều thời gian và công sức để rà soát, tinh chỉnh tọa độ và xóa các hộp rác/dự đoán sai.
    - Khi nâng threshold: Reviewer ít phải xóa box rác do độ chính xác của hộp cao hơn, nhưng reviewer và annotator lại phải bỏ công vẽ thủ công lại từ đầu rất nhiều vật thể bị mô hình bỏ sót.
- Đề xuất một quy tắc box chặt:
  - Quy tắc Bounding Box Ôm Sát: Bounding box phải bao trọn tất cả các điểm cực trị ngoài cùng nhìn thấy được của vật thể (`x_min, y_min, x_max, y_max`). Khoảng cách sai số giữa các cạnh của hộp và mép ngoài cùng của vật thể không được vượt quá 2 pixel (hoặc <= 3% kích thước cạnh tương ứng). Tuyệt đối không để khoảng đệm nền thừa quá lớn và không được cắt lẹm vào phần nhìn thấy của vật thể.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - *Guideline cần quy định rõ:*
    1. Ngưỡng nhìn thấy tối thiểu :Quy định rõ tỷ lệ nhìn thấy tối thiểu để một đối tượng bị che khuất hoặc bị cắt mép được gán nhãn.
    2. Quy cách vẽ box khi che khuất: Vẽ box chỉ bao quanh phần nhìn thấy được hay ước lượng cả phần bị che khuất.
    3. Thuộc tính bắt buộc:Gán thêm cờ nhãn `is_occluded=True` hoặc `is_truncated=True` để phục vụ huấn luyện chuyên sâu.
  - *Escalation:* Khi vật thể bị che khuất quá nặng hoặc bị cắt mép đến mức không thể xác định chắc chắn phân loại dựa trên ngữ cảnh thị giác (ví dụ: khuôn nướng/đĩa úp trên bàn là `bowl` hay khay nướng), annotator không được tự ý phỏng đoán mà phải gắn cờ escalation gửi Reviewer / Lead để thống nhất quy tắc hoặc đánh dấu `ambiguous` / `difficult`.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
  `{"instance_id": "kitchen-001", "class_name": "person", "score": 0.899318, "polygon_point_count": 348, "polygon_xy": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], "...", [456.0, 71.0], [455.0, 70.0]]}`

- Polygon bổ sung chi tiết gì so với box?
  - Bounding box chỉ là một hình chữ nhật bao quanh 4 cạnh, luôn bao hàm cả các mảng điểm ảnh nền và các chi tiết của vật thể lân cận lọt vào góc hộp 
  - Polygon cung cấp đường biên hình học khép kín bám sát từng đường cong, nếp gấp quần áo và chu vi thực tế của đối tượng. Nó phân định chính xác tuyệt đối điểm ảnh nào thuộc về thực thể và điểm ảnh nào là nền, cho phép tính toán diện tích thực tế, hình thái học và tương tác không gian chính xác giữa các đối tượng.
- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` là mã định danh cục bộ duy nhất cho từng cá thể/thể hiện vật thể độc lập trong cùng một bức ảnh. Nó giúp phân biệt rõ ràng giữa hai hay nhiều đối tượng cùng thuộc một lớp ngữ nghĩa.
  - `instance_id` không phải là:
    1. Không phải `class_id` 
    2. Không phải tracking ID
    3. Không phải database global ID.
- Đề xuất một quy tắc biên mask:
  - Quy tắc Bám Biên Đa Giác Chính Xác Pixel: Biên đa giác phải bám sát đường viền tự nhiên thực tế của vật thể với sai số không vượt quá 1 - 2 pixel. Tuyệt đối không được cắt lấn vào thân vật thể và không được bao gồm viền nền rộng quá 2 pixel. Đối với các vật thể có lỗ rỗng hoặc khoảng trống xuyên thấu, bắt buộc phải tạo vùng rỗng, không được phủ kín mask lên phần hậu cảnh lọt qua lỗ.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định rõ:
    1. Ranh giới tiếp xúc chồng lấn : Khi hai vật thể nằm đè lên nhau, guideline phải quy định đường ranh giới tiếp xúc thuộc về vật thể nằm ở lớp trên (quy tắc z-order/depth: vật thể phía trước sở hữu biên tiếp xúc, vật thể phía sau bị trừ phần giao).
    2. Xử lý bóng đổ và vùng nhòe : Quy định rõ bóng đổ không được tính vào mask của vật thể; biên mask phải dừng lại tại điểm tiếp xúc cơ học thực giữa vật thể và bề mặt.
    3. Che khuất chia cắt : Khi một vật thể bị vật thể khác che ngang chia thành hai mảnh nhìn thấy rời rạc, guideline phải quy định xem gán là 1 instance duy nhất có cấu trúc Multi-polygon hay chia làm 2 instance riêng.
  - Escalation: Khi biên giới của vật thể bị nhòe hoàn toàn do thiếu sáng, cháy sáng hoặc mờ nét chuyển động đến mức mắt người không thể phân định được ranh giới pixel với độ tin cậy tối thiểu, annotator phải gắn cờ escalation gửi lên Reviewer/Lead để hội ý hoặc đánh dấu vùng mơ hồ.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn danh mục (`class_id`, `class_name`, `taxonomy_name`) cho toàn bộ bức ảnh (single-label). | Ảnh `traffic` có nhiều chủ thể (xe buýt lớn, ô tô con, người đi bộ) nhưng mô hình bị ép gán 1 nhãn duy nhất và dự đoán nhầm thành `cab` (0.51). | Đọc kỹ thứ tự ưu tiên trong guideline (chọn chủ thể chiếm diện tích lớn nhất ở tiền cảnh); gán đúng 1 nhãn chuẩn; nếu ảnh quá mơ hồ/không có chủ thể chính thì gắn cờ escalation. | Kiểm tra nhãn được chọn có tuân thủ đúng cây quyết định và thứ tự ưu tiên của guideline hay không; đối chiếu xem có bị nhầm lẫn giữa các lớp tương đồng (như cab vs car vs minibus) hay không. |
| Phát hiện vật thể | Danh sách các bounding box tọa độ pixel `[x_min, y_min, x_max, y_max]` kèm `class_name` cho từng vật thể nhìn thấy. | Mô hình bỏ sót nhiều xoong chảo treo trên tường trong ảnh `kitchen`; phát hiện nhầm một cánh tay bị cắt mép sát lề trái là `person` (0.61); nhầm lẫn giữa khuôn nướng/đĩa với `bowl`. | Vẽ bổ sung các bounding box cho vật thể bị bỏ sót; căn chỉnh 4 cạnh ôm sát biên vật thể (tight box); xóa các box dự đoán sai; gắn cờ `is_occluded`/`is_truncated` theo đúng ngưỡng nhìn thấy trong guideline. | Kiểm tra độ chặt chẽ của box (sai số <= 2px, không cắt lẹm thân vật thể); kiểm tra tỷ lệ sót vật thể (False Negatives); kiểm tra các trường hợp che khuất/cắt mép xem có gán nhãn đúng quy định không. |
| Instance segmentation | Mã `instance_id` duy nhất kèm danh sách tọa độ đa giác khép kín `polygon_xy = [[x, y], ...]` bám sát biên pixel cho từng thực thể. | Mask bàn ăn (`dining table`) bị lem và thủng lỗ không đều quanh đồ vật trên bàn; chùm thảo mộc khô treo góc trái bị nhận nhầm thành `potted plant` (0.63); các dụng cụ treo tường bị vẽ biên thô. | Dùng công cụ polygon/brush vẽ đường viền bám sát chu vi thực tế từng vật thể; khoét lỗ rỗng (cutout) cho các khoảng trống nền bên trong; tách riêng các instance tiếp xúc nhau; gán đúng `instance_id` duy nhất. | Phóng to kiểm tra độ chính xác của đường biên đa giác (sai số <= 1-2px); kiểm tra xem có bị dính liền hai instance khác nhau vào cùng một mask không; kiểm tra các chi tiết nhỏ (quai ấm, chân bàn, mép quần áo) có bị cắt đứt không. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  Tuân thủ nghiêm ngặt nguyên tắc tối thiểu hóa dữ liệu và bảo mật quyền riêng tư: Tuyệt đối không tải lên bất kì nơi nào các hình ảnh chứa thông tin định danh cá nhân, dữ liệu nội bộ, bí mật kinh doanh của đối tác và doanh nghiệp, hoặc dữ liệu không thuộc phạm vi cho phép của dự án.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  Giảng viên / Mentor phụ trách học phần (hoặc Project Manager / Data Protection Officer của dự án) qua kênh liên lạc chính thức, đồng thời lập tức cô lập tệp dữ liệu nghi vấn và không sao chép, phát tán ra ngoài.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
