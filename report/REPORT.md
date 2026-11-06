# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn  
  
**Ngày chạy: 06-07/11/2026**  
  
**Runtime Colab:** GPU (T4; Device: cuda)  
   
**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145  
  
**Checkpoint:** ==yolo11n-cls.pt==, ==yolo11n.pt==, ==yolo11n-seg.pt==  
  
**Ngưỡng detection/segmentation dùng khi tạo JSON:** ==0.35== (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)  
  
**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi (Bổ sung ô đọc/lọc JSON để kiểm tra record của ảnh `kitchen`;  
giữ nguyên checkpoint và ngưỡng tạo JSON `0.35`.)  
  
> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành ==REPORT.md==, rồi **upload** vào panel Files   
> 
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng ==day1_lab_outputs/==.   
> 
> Giải nén và đưa vào ==report/== trên GitHub, rồi nộp **link repo** trên LMS.   
> 
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;   
> 
> họ tên/MSSV chỉ nằm trong tên repository.  
  

**Từ cần nhớ (đọc nhanh trước khi viết)**  

| Từ | Nghĩa ngắn |
| ------------- | ------------------------------------------------------------- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model .pt đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |
  
  
## 1. Phân loại ảnh  
  
**Mở file:** ==day1_lab_outputs/classification_predictions.json== và ==visuals/classification_top5.png==.  
Chỉ dùng ảnh ==sample_id = traffic==, lấy dòng ==rank = 1==.  
  
1. Chép record hạng 1: ==class_id=== 468 ; ==class_name=== cab ; ==rank===1 ; ==score=== 0.510915 ; ==taxonomy_name=== ImageNet-1K  
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy? Record mô tả **toàn bộ ảnh**, vì checkpoint yolo11n-cls.pt thực hiện image classification và xếp hạng các lớp ImageNet-1K cho cả khung hình. cab là prediction đứng đầu với score 0.510915; record không cung cấp vị trí của taxi và chưa chứng minh đây là ground truth đúng.  
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint? Danh sách (class list) đi kèm với taxonomy/checkpoint, bởi vì khi huấn luyện một mô hình học sâu (như YOLO), phải định nghĩa trước taxonomy - ví dụ: ImageNet-1K có 1000 lớp, COCO-80 có 80 lớp). Trong quá trình huấn luyện, mô hình học cách phân loại các đặc trưng của ảnh vào các chỉ số lớp (class ID) tương ứng trong danh sách này. Khi lưu trữ mô hình thành tệp checkpoint (.pt), danh sách tên lớp (names) sẽ được đóng gói kèm theo cấu trúc của checkpoint đó để phục vụ cho inference sau này  
4. Vì sao cần ghi cả ==class_id==, ==class_name== và ==taxonomy_name==, không chỉ ghi mỗi tên lớp? Đầu tiên là tránh gây xung đột và hiểu lầm ý nghĩa (vì ở các taxonomy khác nhau, thì định nghĩa các tên khác nhau). Tối ưu hiệu năng máy tính (làm việc riêng với class id), cuối cùng là hỗ trợ ngôn ngữ và chuẩn hóa dữ liệu (classname có vai trò dịch ngược từ các id sang tên để người đọc có thể hiểu được)  
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần quy định rõ nguyên tắc ưu tiên để chọn nhãn - Dominance Rule -  chọn ảnh theo diện tích và mức độ nổi bật, Class Hierarchy - chọn ảnh theo nhu cầu ưu tiên (người > vật> phương tiện) nếu các vật thể có diện tích ngang bằng nhau, Context Rule -  Quy định khi nào dán nhãn chung ( street - thay vì dán nhãn tất cả mọi vật thể xuất hiện trên đường), Escalation khi gặp các ca khó, vd : các vật thể bị chen lấn và đè lên nhau  
6. Vì sao không được lấy ==score== của model làm ground truth? Không được, vì bản chất của khác nhau, một cái là mốc chuẩn để dạy model học, một cái là độ tin cậy của model tại thời điểm model học, có thể đoán đúng nhưng score thấp vì e dè, hoặc score cao nhưng labelling sai  
  
## 2. Phát hiện vật thể  
  
**Mở file:** ==day1_lab_outputs/detection_predictions.json== và ==visuals/detection_predictions.png==.  
Chỉ dùng ảnh ==sample_id = kitchen==. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh ==0.20== / ==0.35== / ==0.60== trong notebook.  
  
1. Chép một record: ==class_name=== person  ; ==score===0.912625 ; ==bbox_xyxy===[385.33, 69.24, 498.92, 348.92] ; ==bbox_width===113.58 ; ==bbox_height===279.68  
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Góc trên-trái (Top-Left): Có tọa độ là (385.33, 69.24). Điều này có nghĩa là điểm bắt đầu của hộp nằm cách biên bên trái của bức ảnh 385.33 pixel và cách biên phía trên của bức ảnh 69.24 pixel. **G**óc dưới-phải (Bottom-Right): Có tọa độ là (498.92, 348.92). Nghĩa là điểm kết thúc của hộp nằm cách biên bên trái của bức ảnh 498.92 pixel và cách biên phía trên của bức ảnh 348.92 pixel. 	Chiều rộng (**Width**): **113.58 pixel** (khoảng cách nằm ngang từ rìa bên trái sang rìa bên phải của người này). Chiều cao (**Height**): **279.68 pixel** (khoảng cách thẳng đứng từ đỉnh đầu xuống tới ngang lưng/chân của người này trong khung hình).  
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? 0.20 có 17 prediction, nhạy nên predict những vật thể nhỏ, còn 0.60 predict ít và chắc chắn hơn những ngưỡng còn lại  
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì? Người reviewer phải lọc và xóa bỏ các dự đoán sai (false positive), vì khi ngưỡng thấp, các khu vực nhiễu hoặc vật thể không liên quan rất dễ bị mô hình nhận diện nhầm (ví dụ: nhận nhầm họa tiết trên tường thành cái bát hoặc cái cốc), reviewer phải click thủ công để xóa các hộp bao sai này. Xử lý overlapping bouding boxes, và tốn nhiều thời gian hơn  
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này)  Mỗi vật thể phân biệt được (**instance**) có một hộp bao riêng, vẽ sát biên (**tight bounding box**) và bao trọn phần nhìn thấy; khi bị che khuất (**occlusion**), không suy đoán phần bị ẩn; khi bị cắt ở mép ảnh (**truncation**), dừng hộp tại mép ảnh; khi các vật thể chồng chéo (**overlap**), vẫn tách hộp cho từng vật thể nếu phân biệt được; nếu không xác định được ranh giới, chuyển người kiểm tra (**reviewer**) hoặc Lab Coach xử lý (**escalation**), đồng thời áp dụng quy ước nhất quán trên mọi ảnh.  
6. Trên ảnh ==kitchen==, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao? **(oven 0.69) bị cắt ở mép trái ảnh — truncation**. Tôi sẽ đưa vào **guideline** quy tắc: vẽ hộp sát phần lò còn nhìn thấy, dừng tại mép ảnh và không suy đoán phần nằm ngoài khung hình. Vùng được model dự đoán là person ở mép trái ảnh chỉ lộ một phần cơ thể (**severe truncation**), khó xác định phạm vi hộp. Nếu guideline chưa hướng dẫn trường hợp này, sẽ hỏi Lab Coach (**escalation**) để thống nhất có gán nhãn hay không và cách vẽ hộp, tránh mỗi annotator tự xử lý khác nhau  
  
## 3. Phân đoạn theo từng đối tượng  
  
**Mở file:** ==day1_lab_outputs/segmentation_predictions.json== và ==visuals/segmentation_prediction.png==.  
Chỉ dùng ảnh ==sample_id = kitchen==. Chọn **một** record.  
  
1. Chép một record: ==instance_id=== kitchen-002 ; ==class_name===person  ; ==score===0.90 ; số điểm polygon===polygon_point_count===122 ; 3–5 điểm đầu của ==polygon_xy===[[453.0, 203.0], [452.0, 204.0], [451.0, 204.0], [450.0, 205.0], [449.0, 205.0]]  
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể? Bounding box chỉ là một hình chữ nhật bao quanh vật thể, vô tình chứa cả các điểm ảnh thuộc về nền, Polygon sẽ bám sát theo đường biên thực tế của vật thể, giúp xác định các hình dạng phức tạp hơn, ngoài ra Polygon hỗ trợ tính toán diện tích thực tế dựa trên số lượng và tọa độ pixel nằm trong đa giác, thay vì diện tích bao quanh hình chữ nhật, cuối cùng là hạn chế overlapping ( trong hình là khi bowl, open, dining tablet đè lên nhau) - polygon sẽ giúp phân định ranh giới rõ ràng của từng object  
3. ==instance_id== dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào? Instance_id nhằm để phân biệt vật thể riêng biệt trong cùng tấm ảnh, ngay cả khi chúng cùng thuộc 1 class, Class_Id là mã định danh chung cho cả một danh mục (vd : person là 0,…) trong khi Instance_id là độc nhất cho mỗi cá thể riêng lẻ trong ảnh ,Tracking ID là mã định danh dùng để theo vết một vật thể xuyên suốt qua nhiều khung hình (video). Ví dụ: vật thể có ID 001 ở khung hình thứ nhất vẫn phải giữ nguyên ID 001 ở khung hình thứ 100 nếu đó là cùng một thực thể chuyển động.  
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo -  Để tạo ra dữ liệu gán nhãn đa giác (polygon mask) đạt tiêu chuẩn chất lượng cao cho mô hình phân đoạn đối tượng, người gán nhãn cần tuân thủ nghiêm ngặt một quy trình thống nhất. Đầu tiên, các điểm đa giác phải được vẽ bám cực kỳ sát vào ranh giới thực tế của vật thể, đảm bảo không để lộ các pixel nền bên trong đa giác và không lấn quá 2 pixel ra ngoài. Người gán nhãn cần chủ động điều chỉnh mật độ điểm vẽ linh hoạt: tăng mật độ điểm neo thật dày để bo mịn các đường cong phức tạp, đồng thời chỉ đặt các điểm ở góc gãy để giữ độ thẳng cho các cạnh tuyến tính. Đối với các trường hợp vật thể bị che khuất một phần bởi chướng ngại vật phía trước, quy tắc bắt buộc là chỉ vẽ đa giác bao quanh phần bề mặt thực tế nhìn thấy được mà không tự ý suy đoán vẽ bù thêm. Cuối cùng, người gán nhãn cần phóng to vùng làm việc tối thiểu từ 150% đến 200% để kiểm soát chính xác từng chi tiết biên, tránh tối đa việc gán sót hoặc thừa các pixel biên của đối tượng.  
5. Trên ảnh ==kitchen==, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao? - Khu vực bàn bếp gỗ ở trung tâm và các vùng xung quanh  Hai vật chạm nhau và chồng lấn cực mạnh: Khu vực các khay/bát ở góc dưới bên trái (bowl chồng lên nhau trên bàn). Đặc biệt là chiếc bát bột và cây cán bột (rolling pin chạm sát và đè lên nhau), ranh giới giữa chúng rất mờ và chồng chéo, 	Vật thể bị che khuất (Occlusion): Chiếc lò nướng (oven màu tím bên trái) bị chiếc bàn gỗ và các dụng cụ trên bàn che khuất gần một nửa phần thân dưới. - Sử dụng guideline: đối với các vật thể bị che khuất 1 phần hay các box cắt vào nhau thì trong guildeline có thể quy định: “không được suy đoán phân bị khuất”, chỉ cần bám theo quy tắc. - Sử dụng escalation: đối với các bowl bị che khuất hoàn toàn, khả năng nhận diện bị biến mất gười gán nhãn (Annotator) không được phép tự ý "bịa" ra đường biên để vẽ bừa. Việc tự ý vẽ sẽ làm sai lệch chất lượng dữ liệu huấn luyện. Do đó, cần phải Escalation (gửi ảnh kèm câu hỏi lên Mentor/Lab Coach hoặc Khách hàng) để họ quyết định xem có gán nhãn những chiếc chảo bị mờ đó không, hoặc cung cấp ảnh gốc có ánh sáng tốt hơn.   
  
## 4. Vòng đời và kiểm tra chất lượng  
  
Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:  
  
==ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework==  
  
Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):  

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh | Là nhãn duy nhất gán cho toàn bộ bức ảnh (vd: Cab, minibus). Không có tọa độ hay hình vẽ | Các cảnh phức tạp chứa quá nhiều vật thể (như traffic có cả xe buýt, xe con, người đi bộ) khiến việc chọn một nhãn duy nhất mang tính chủ quan cao. | Xem toàn bộ ảnh và chọn một nhãn lớp phù hợp nhất trong danh sách taxonomy ImageNet-1K dựa theo đối tượng nổi bật nhất. | Kiểm tra xem nhãn được chọn có phản ánh đúng đối tượng chủ đạo của bức ảnh hay không, đảm bảo tính nhất quán giữa các ảnh tương tự |
| Phát hiện vật thể | Gồm nhãn lớp kèm theo tọa độ hộp giới hạn dạng hình chữ nhật bbox_xyxy (đơn vị pixel: [x_min, y_min, x_max, y_max]). | Các vật thể nhỏ, mờ ở phía xa bị bỏ sót hoặc các vật thể bị che khuất một phần (như lò nướng oven bị bàn bếp che) khó xác định ranh giới hộp hoàn chỉnh. | Vẽ một hình chữ nhật bao quanh khít nhất có thể từng đối tượng đơn lẻ và chọn nhãn lớp tương ứng (ví dụ: person, bowl). | Kiểm tra xem hộp vẽ có quá rộng hay quá hẹp không, có bỏ sót vật thể nào không, và nhãn lớp được gán đã chuẩn xác chưa. |
| Instance segmentation | Gồm nhãn lớp, mã thực thể riêng biệt instance_id và một đa giác bám sát biên polygon_xy (danh sách các điểm tọa độ [x, y]). | Các vật thể chạm sát nhau, chồng lấn mạnh (như bát đĩa trên bàn) hoặc bị bóng tối che khuất làm mờ ranh giới (như chiếc chảo treo phía trên). | Phóng to ảnh, chấm các điểm neo đa giác chạy dọc sát theo đường biên thực tế nhìn thấy của từng thực thể, tránh gán nhãn phần bị che khuất. | Kiểm tra độ mịn và độ khít của đa giác so với biên thực tế của vật thể, đảm bảo các thực thể chồng lấn được tách biệt rõ ràng. |
  
  
## 5. An toàn dữ liệu  
  
1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)  - Chỉ sử dụng các ảnh mẫu COCO công khai có sẵn được cung cấp trong bài lab, tuyệt đối không tải lên hoặc sử dụng bất kỳ hình ảnh cá nhân hay dữ liệu nội bộ riêng tư nào.  
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai? - 	Nếu phát hiện hình ảnh hoặc dữ liệu không đúng phạm vi quy định hoặc nhạy cảm, tôi sẽ dừng ngay lập tức và tiến hành báo cáo (Escalation) cho Lab Coach hoặc giảng viên hướng dẫn để được xử lý.  
  
## 6. Checklist trước khi nộp link repo  
  
Kiểm tra trên GitHub trong thư mục ==report/==:  
  
- [x] ==REPORT.md== (file này đã điền)  
- [x] ==day1_lab_outputs/classification_predictions.json==  
- [x] ==day1_lab_outputs/detection_predictions.json==  
- [x] ==day1_lab_outputs/segmentation_predictions.json==  
- [x] ==day1_lab_outputs/IMAGE_ATTRIBUTION.md==  
- [x] ==day1_lab_outputs/visuals/classification_top5.png==  
- [x] ==day1_lab_outputs/visuals/detection_predictions.png==  
- [x] ==day1_lab_outputs/visuals/segmentation_prediction.png==  
- [x] Ô validation cuối notebook báo ==PASS==  
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output  
- [x] Đã nộp **link repository** trên LMS  
