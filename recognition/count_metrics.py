from collections import defaultdict

def bb_intersection_over_union(box_a, box_b):
    '''
    box_a - первая ограничительная рамка
    box_b - вторая ограничительная рамка
    '''
    a_x = max(box_a[0], box_b[0])
    a_y = max(box_a[1], box_b[1])
    b_x = max(box_a[2], box_b[2])
    b_y = max(box_a[3], box_b[3])
    inter_area = max(0, b_x - a_x + 1) * max(0, b_y - a_y + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    IOU = inter_area / float(box_a_area + box_b_area - inter_area)
    return IOU

def find_best_iou(dd, iou_results):
    '''
    Использовать функцию, если есть несколько ограничительных рамок, 
    и нужно выбрать наиболее подходящую, а другие удалить
    '''
    for k, v in dd.items():
        max_iou = 0
        values = list(v)
        for i, value in enumerate(values):
            if iou_results[value][1] > max_iou:
                if i != 0:
                    del iou_results[values[i-1]]
            else:
                del iou_results[values[i]]
                max_iou = iou_results[value][1]

    return iou_results

def count_all_needed_indicators(iou_results, correct_boxes, detected_boxes):
	not_find_face = len(correct_boxes) - len(iou_results.values())
	fp = len(detected_boxes) - len(iou_results.values())
	iou_lst = [iou_results[item][1] for item in iou_results]
	return not_find_face, fp, iou_lst


def find_iou_for_all_boxes(correct_boxes, detected_boxes):
	"""
	Parameters:
		 correct_boxes(list of list): labeled bounding boxes
		 detected_boxes(list of list): bounding boxes that return NN
	Returns:
		not_find_face(int): number of missing faces
		fp(int): the number of objects(faces) found where there are none
		iou_lst(int): intersection over Union all found faces in the image
	"""
	iou_results = {}
	for true_box in correct_boxes:
		for detected_box in detected_boxes:
			iou = bb_intersection_over_union(true_box, detected_box)
			if str(true_box) in iou_results:
				if iou_results[str(true_box)][1] < iou:
					iou_results[str(true_box)] = [detected_box, iou]
			else:
				iou_results[str(true_box)] = [detected_box, iou]

		# remove false positive result
		if iou_results:
			last_element_in_dct = iou_results[list(iou_results)[-1]]
			if last_element_in_dct[1] == 0.0:
				del iou_results[list(iou_results)[-1]]

	dd = defaultdict(set)

	for key, value in iou_results.items():
		dd[str(value[0])].add(key)
	dd = {k: v for k, v in dd.items() if len(v) > 1}
	if dd:
		iou_results = find_best_iou_for_many(dd, iou_results)
		not_find_face, fp, iou_lst = count_all_needed_indicators(iou_results, correct_boxes, detected_boxes)
		return not_find_face, fp, iou_lst
	else:
		not_find_face, fp, iou_lst = count_all_needed_indicators(iou_results, correct_boxes, detected_boxes)
		return not_find_face, fp, iou_lst


def count_precision_and_recall(tp, fn, fp):
	print('precision: '+str(tp/(tp+fp)))
	print('recall: '+str(tp/(tp+fn)))
    
