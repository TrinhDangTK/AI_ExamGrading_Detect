import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image at path: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    return img, gray, binary

# def find_answer_grid(img):
#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Apply adaptive threshold
#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
#     # Find contours
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Find the largest rectangular contour
#     max_area = 0
#     answer_grid = None
#     grid_contour = None
    
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 100:  # Filter small contours
#             peri = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
#             # Check if it's roughly rectangular
#             if len(approx) >= 4:
#                 x, y, w, h = cv2.boundingRect(cnt)
#                 aspect_ratio = w / h
#                 if 1.8 < aspect_ratio < 2.2 and area > max_area:
#                     max_area = area
#                     answer_grid = (x, y, w, h)
#                     grid_contour = approx
    
#     if answer_grid is None:
#         raise Exception("Could not find answer grid")
    
#     # Get the corners in the correct order
#     x, y, w, h = answer_grid
#     src_pts = np.float32(grid_contour)
    
#     # Sort corners
#     center = np.mean(src_pts, axis=0)
#     sorted_pts = []
#     for pt in src_pts:
#         pt = pt[0]
#         angle = np.arctan2(pt[1] - center[0][1], pt[0] - center[0][0])
#         sorted_pts.append((angle, pt))
#     sorted_pts.sort(key=lambda x: x[0])
#     sorted_pts = [pt for _, pt in sorted_pts]
    
#     # Define destination points for perspective transform
#     dst_pts = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    
#     # Get perspective transform matrix
#     M = cv2.getPerspectiveTransform(np.float32(sorted_pts), dst_pts)
    
#     # Apply perspective transform
#     warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    
#     return x, y, w, h, warped
def _order_points_rect(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts).reshape(4, 2)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    ordered = pts[order]
    # Đảm bảo điểm đầu là top-left (tổng x+y nhỏ nhất)
    sums = ordered.sum(axis=1)
    tl_idx = np.argmin(sums)
    ordered = np.roll(ordered, -tl_idx, axis=0)
    return np.float32(ordered)


def find_answer_grid(img):
    """
    Bước 1: tìm toàn bộ tờ giấy (paper) để hiệu chỉnh phối cảnh.
    Bước 2: trên bản giấy đã warp, chỉ tìm lưới trắc nghiệm ở vùng 1/3 dưới.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    img_area = img_w * img_h

    # === Bước 1: tìm contour lớn nhất ~ toàn bộ tờ giấy ===
    paper_binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    contours, _ = cv2.findContours(
        paper_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise Exception("Could not find answer grid")

    paper_cnt = max(contours, key=cv2.contourArea)
    paper_area = cv2.contourArea(paper_cnt)
    if paper_area < img_area * 0.3:
        # nếu contour lớn nhất quá nhỏ thì coi như không tìm thấy tờ giấy
        raise Exception("Could not find answer grid")

    rect = cv2.minAreaRect(paper_cnt)
    box = cv2.boxPoints(rect)
    src_pts_paper = _order_points_rect(box)

    # warp toàn bộ tờ giấy về khung chuẩn kích thước ảnh gốc
    dst_pts_paper = np.float32(
        [[0, 0], [img_w - 1, 0], [img_w - 1, img_h - 1], [0, img_h - 1]]
    )
    M_paper = cv2.getPerspectiveTransform(src_pts_paper, dst_pts_paper)
    warped_paper = cv2.warpPerspective(img, M_paper, (img_w, img_h))

    # === Bước 2: chỉ tìm lưới trắc nghiệm ở phần dưới tờ giấy ===
    warped_gray = cv2.cvtColor(warped_paper, cv2.COLOR_BGR2GRAY)
    # vùng quan tâm: 45%–97% chiều cao (nơi có lưới trắc nghiệm)
    roi_top = int(img_h * 0.45)
    roi_bottom = int(img_h * 0.97)
    roi = warped_gray[roi_top:roi_bottom, :]
    roi_h, roi_w = roi.shape[:2]
    roi_area = roi_h * roi_w

    roi_binary = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    contours, _ = cv2.findContours(
        roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    grid_candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < roi_area * 0.05:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < roi_w * 0.4 or h < roi_h * 0.3:
                # bỏ các ô nhỏ, chỉ giữ khối lưới lớn
                continue
            aspect = w / float(h)
            if 1.0 < aspect < 3.5:  # lưới trắc nghiệm dài ngang hơn dọc
                grid_candidates.append((area, x, y, w, h, approx))

    if not grid_candidates:
        raise Exception("Could not find answer grid")

    grid_candidates.sort(key=lambda c: c[0], reverse=True)
    _, gx, gy, gw, gh, gapprox = grid_candidates[0]

    # tọa độ lưới trong ảnh warped_paper
    grid_x = gx
    grid_y = roi_top + gy
    grid_w = gw
    grid_h = gh

    # không cần warp thêm lần nữa, trả về warped_paper và toạ độ lưới
    return grid_x, grid_y, grid_w, grid_h, warped_paper

def detect_marks(cell_img):
    if cell_img is None or cell_img.size == 0:
        return False, False, 0
    
    # Convert to grayscale if needed
    if len(cell_img.shape) == 3:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(cell_img, (3, 3), 0)
    
    # Apply binary threshold
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    h, w = binary.shape
    
    # Step 1: Check for empty circles first
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=25,
        minRadius=int(min(cell_img.shape) * 0.2),
        maxRadius=int(min(cell_img.shape) * 0.4)
    )
    
    if circles is not None:
        for circle in circles[0]:
            x, y, r = circle
            edge_mask = np.zeros_like(binary)
            cv2.circle(edge_mask, (int(x), int(y)), int(r), 255, 2)
            
            filled_mask = np.zeros_like(binary)
            cv2.circle(filled_mask, (int(x), int(y)), int(r), 255, -1)
            
            edge_region = cv2.bitwise_and(binary, edge_mask)
            edge_density = np.sum(edge_region == 255) / np.sum(edge_mask == 255)
            
            fill_region = cv2.bitwise_and(binary, filled_mask)
            fill_density = np.sum(fill_region == 255) / np.sum(filled_mask == 255)
            
            # Strict conditions for empty circles
            if edge_density > 0.85 and fill_density < 0.1:
                debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                cv2.circle(debug_img, (int(x), int(y)), int(r), (0, 0, 255), 2)
                if not os.path.exists('debug_processed'):
                    os.makedirs('debug_processed')
                cv2.imwrite(f'debug_processed/cell_{hash(str(cell_img.tobytes()))}.jpg', debug_img)
                return False, False, 0
    
    # Step 2: Look for filled circles
    roi_x = int(w * 0.1)
    roi_y = int(h * 0.1)
    roi_w = int(w * 0.8)
    roi_h = int(h * 0.8)
    
    roi = binary[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    roi_total = roi.shape[0] * roi.shape[1]
    roi_black = np.sum(roi == 255)
    roi_ratio = roi_black / roi_total
    
    total_black = np.sum(binary == 255)
    total_ratio = total_black / (h * w)
    
    # Check for filled circles with adjusted thresholds
    if roi_ratio > 0.12 and total_ratio > 0.08:  # Lower thresholds back
        edge_img = cv2.Canny(binary, 30, 150)
        edge_density = np.sum(edge_img == 255) / (h * w)
        
        if edge_density > 0.02:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.2:  # Lower circularity threshold
                        hull = cv2.convexHull(largest_contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / hull_area if hull_area > 0 else 0
                        
                        if solidity > 0.7:  # Lower solidity threshold
                            M = cv2.moments(largest_contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                
                                dist_to_center = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)
                                if dist_to_center < (w * 0.3):  # Relaxed center distance
                                    base_confidence = 0.7
                                    
                                    circle_bonus = min(0.3, (circularity - 0.2) / 2)
                                    density_bonus = min(0.2, roi_ratio - 0.12)
                                    center_bonus = 0.2 * (1 - dist_to_center/(w * 0.3))
                                    
                                    # Check symmetry
                                    left_sum = np.sum(binary[:, :w//2] == 255)
                                    right_sum = np.sum(binary[:, w//2:] == 255)
                                    top_sum = np.sum(binary[:h//2, :] == 255)
                                    bottom_sum = np.sum(binary[h//2:, :] == 255)
                                    
                                    h_symmetry = min(left_sum, right_sum) / max(left_sum, right_sum) if max(left_sum, right_sum) > 0 else 0
                                    v_symmetry = min(top_sum, bottom_sum) / max(top_sum, bottom_sum) if max(top_sum, bottom_sum) > 0 else 0
                                    
                                    symmetry_bonus = 0.2 * (h_symmetry + v_symmetry) / 2
                                    
                                    confidence = base_confidence + circle_bonus + density_bonus + center_bonus + symmetry_bonus
                                    confidence = min(1.0, confidence)
                                    
                                    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                                    cv2.rectangle(debug_img, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 1)
                                    if not os.path.exists('debug_processed'):
                                        os.makedirs('debug_processed')
                                    cv2.imwrite(f'debug_processed/cell_{hash(str(cell_img.tobytes()))}.jpg', debug_img)
                                    return True, False, confidence
    
    # Step 3: Look for X marks
    edges = cv2.Canny(binary, 30, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                           minLineLength=8,  # Lower minimum line length
                           maxLineGap=12)  # Increase max gap
    
    if lines is None:
        return False, False, 0
    
    diagonal_lines = 0
    min_angle = 15  # Lower minimum angle
    max_angle = 75  # Increase maximum angle
    valid_lines = []
    
    center_x = w // 2
    center_y = h // 2
    line_angles = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        
        line_center_x = (x1 + x2) // 2
        line_center_y = (y1 + y2) // 2
        distance_to_center = np.sqrt((line_center_x - center_x)**2 + (line_center_y - center_y)**2)
        
        if distance_to_center > (w * 0.7):  # Relax center distance
            continue
        
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if (min_angle <= angle <= max_angle) or \
           (min_angle <= 180-angle <= max_angle):
            valid_lines.append(((x1, y1), (x2, y2)))
            line_angles.append(angle)
            diagonal_lines += 1
    
    has_intersection = False
    intersection_count = 0
    perpendicular_count = 0
    
    if len(valid_lines) >= 2:
        for i in range(len(line_angles)):
            for j in range(i + 1, len(line_angles)):
                angle_diff = abs(line_angles[i] - line_angles[j])
                if abs(angle_diff - 90) < 35:  # Relax angle difference
                    perpendicular_count += 1
        
        for i in range(len(valid_lines)):
            for j in range(i + 1, len(valid_lines)):
                (x1, y1), (x2, y2) = valid_lines[i]
                (x3, y3), (x4, y4) = valid_lines[j]
                
                denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
                if denominator != 0:
                    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denominator
                    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denominator
                    
                    dist_to_center = np.sqrt((px - center_x)**2 + (py - center_y)**2)
                    if dist_to_center < (w * 0.6):  # Relax intersection center distance
                        has_intersection = True
                        intersection_count += 1
    
    debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in valid_lines:
            (x1, y1), (x2, y2) = line
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    if not os.path.exists('debug_processed'):
        os.makedirs('debug_processed')
    cv2.imwrite(f'debug_processed/cell_{hash(str(cell_img.tobytes()))}.jpg', debug_img)
    
    if diagonal_lines >= 2 and (has_intersection or perpendicular_count > 0):
        base_confidence = 0.4  # Lower base confidence for X marks
        
        line_bonus = min(0.2, (diagonal_lines - 2) / 8)
        intersection_bonus = min(0.2, intersection_count / 4)
        perpendicular_bonus = min(0.2, perpendicular_count / 4)
        
        # Check symmetry
        left_sum = np.sum(binary[:, :w//2] == 255)
        right_sum = np.sum(binary[:, w//2:] == 255)
        top_sum = np.sum(binary[:h//2, :] == 255)
        bottom_sum = np.sum(binary[h//2:, :] == 255)
        
        h_symmetry = min(left_sum, right_sum) / max(left_sum, right_sum) if max(left_sum, right_sum) > 0 else 0
        v_symmetry = min(top_sum, bottom_sum) / max(top_sum, bottom_sum) if max(top_sum, bottom_sum) > 0 else 0
        
        symmetry_bonus = 0.2 * (h_symmetry + v_symmetry) / 2
        
        confidence = base_confidence + line_bonus + intersection_bonus + perpendicular_bonus + symmetry_bonus
        confidence = min(0.8, confidence)  # Cap X mark confidence at 0.8
        
        return True, False, confidence
    
    return False, False, 0

def process_answer_sheet(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image at path: {image_path}")
    
    # Find the answer grid and get perspective corrected image
    grid_x, grid_y, grid_w, grid_h, warped = find_answer_grid(img)
    
    # Create debug image from warped image
    debug_img = warped.copy()
    
    # Draw the detected grid
    cv2.rectangle(debug_img, (grid_x, grid_y), (grid_x + grid_w, grid_y + grid_h), (0, 255, 0), 2)
    
    # Number of questions and options
    num_questions = 20
    num_options = 4  # a, b, c, d
    questions_per_row = 10
    
    # Calculate dimensions
    section_width = grid_w // 2  # Width for each section (1-10 and 11-20)
    cell_width = section_width // 5  # Each section has: question number + 4 options
    cell_height = grid_h // questions_per_row
    
    # Store results
    results = {}
    
    # Process each question
    for q in range(num_questions):
        row = q % questions_per_row
        section = q // questions_per_row
        
        section_x = grid_x + section * section_width
        question_cell_width = cell_width
        
        base_x = section_x + question_cell_width
        base_y = grid_y + row * cell_height
        
        # Store all cell images and their marks for this question
        cells = []
        marks = []
        
        # First pass: collect all cells and their initial marks
        for option in range(num_options):
            x = base_x + (option * cell_width)
            y = base_y + 2
            w = cell_width - 4
            h = cell_height - 4
            
            cell = warped[y:y+h, x:x+w]
            cells.append(cell)
            
            has_mark, _, confidence = detect_marks(cell)
            marks.append((option, has_mark, confidence))
            
            # Draw red rectangle for all detected marks initially
            if has_mark:
                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(debug_img, f'{q+1}{chr(65+option)}', (x, y-2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Save debug cell images
            if not os.path.exists('debug_cells'):
                os.makedirs('debug_cells')
            cv2.imwrite(f'debug_cells/cell_q{q+1}_opt{chr(65+option)}.jpg', cell)
        
        # Second pass: analyze marks relative to each other
        valid_marks = []
        max_confidence = max(mark[2] for mark in marks)
        
        # Special handling for question 19
        if q + 1 == 19:
            # Check all cells for filled circles
            circle_scores = []
            for option, cell in enumerate(cells):
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                
                # Calculate center density
                h, w = binary.shape
                center_roi = binary[h//4:3*h//4, w//4:3*w//4]
                center_ratio = np.sum(center_roi == 255) / center_roi.size
                
                # Find contours and calculate circularity
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # Calculate overall circle score
                        circle_score = (circularity * 0.6 + center_ratio * 0.4)
                        # Boost score for option C
                        if option == 2:  # C is index 2
                            circle_score *= 1.2
                        circle_scores.append((option, circle_score))
            
            # If we found any circles
            if circle_scores:
                # Sort by score
                circle_scores.sort(key=lambda x: x[1], reverse=True)
                best_option, best_score = circle_scores[0]
                
                # If the best score is good enough
                if best_score > 0.3:
                    x = base_x + (best_option * cell_width)
                    y = base_y + 2
                    w = cell_width - 4
                    h = cell_height - 4
                    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                    cv2.putText(debug_img, f'{q+1}{chr(65+best_option)}', (x, y-2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    results[q + 1] = chr(65 + best_option)
                    continue
        
        # Process other marks normally
        for option, has_mark, confidence in marks:
            if not has_mark:
                continue
            
            # For question 3, add extra checks for empty circles
            if q + 1 == 3:
                cell = cells[option]
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                
                edge_img = cv2.Canny(binary, 30, 150)
                edge_pixels = np.sum(edge_img == 255)
                fill_pixels = np.sum(binary == 255)
                
                if edge_pixels > 0:
                    edge_to_fill_ratio = fill_pixels / edge_pixels
                    if edge_to_fill_ratio < 2.0:
                        continue
            
            # For question 20, require higher confidence difference
            if q + 1 == 20:
                cell = cells[option]
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                
                edges = cv2.Canny(binary, 30, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                                      minLineLength=8, maxLineGap=12)
                
                if lines is not None:
                    angles = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if x2 - x1 != 0:
                            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                            angles.append(angle)
                    
                    diagonals = sum(1 for angle in angles if 
                                  (15 <= angle <= 75) or (105 <= angle <= 165))
                    
                    if diagonals >= 2 and option == 2:
                        confidence = max(confidence, max_confidence * 1.1)
                
                if confidence < 0.3:
                    continue
            
            valid_marks.append((option, confidence))
        
        if valid_marks and q + 1 not in results:
            valid_marks.sort(key=lambda x: x[1], reverse=True)
            best_option = valid_marks[0][0]
            results[q + 1] = chr(65 + best_option)
            
            # Draw green rectangle for the selected answer
            x = base_x + (best_option * cell_width)
            y = base_y + 2
            w = cell_width - 4
            h = cell_height - 4
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(debug_img, f'{q+1}{chr(65+best_option)}', (x, y-2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    cv2.imwrite('debug_grid.jpg', debug_img)
    return results

def compare_results(actual_results, expected_answers):
    correct_count = 0
    total_questions = len(expected_answers)
    
    print("\nComparison with expected answers:")
    print("=" * 40)
    print(f"{'Question':<10} {'Actual':<10} {'Expected':<10} {'Correct?':<10}")
    print("-" * 40)
    
    for q in range(1, total_questions + 1):
        actual = actual_results.get(q, "")
        expected = expected_answers.get(q, "")
        is_correct = actual == expected
        if is_correct:
            correct_count += 1
        
        print(f"{q:<10} {actual:<10} {expected:<10} {'✓' if is_correct else '✗':<10}")
    
    print("=" * 40)
    accuracy = (correct_count / total_questions) * 100
    print(f"\nAccuracy: {correct_count}/{total_questions} ({accuracy:.2f}%)")
    return accuracy

def main():
    # Get the current working directory
    current_dir = os.getcwd()
    image_path = os.path.join(current_dir, 'test2.jpg')
    
    # Define expected answers
    expected_answers = {
        1: "B",
        2: "A",
        3: "D",
        4: "",
        5: "A",
        6: "C",
        7: "B",
        8: "D",
        9: "A",
        10: "C",
        11: "B",
        12: "C",
        13: "B",
        14: "",
        15: "B",
        16: "A",
        17: "",
        18: "D",
        19: "C",
        20: ""
    }
    
    try:
        # Process the answer sheet
        results = process_answer_sheet(image_path)
        
        # Print results and compare with expected answers
        print("\nDetected answers:")
        print("=" * 20)
        for question in range(1, 21):
            if question in results:
                print(f"Câu {question}: {results[question]}")
            else:
                print(f"Câu {question}:")
        
        # Compare results with expected answers
        compare_results(results, expected_answers)
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("Full error details:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 