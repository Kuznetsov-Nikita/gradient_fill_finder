import cv2 as cv
import json
import numpy as np
import os
import pathlib
import sys

max_sum_of_colors = 755

# Градиентный фильтр Лапласа
def laplace_grad(image):
	image = cv.Laplacian(image, cv.CV_32F)
	image_grad = cv.convertScaleAbs(image)
	#cv.imshow("lapalace", image_grad)
	return image_grad

# Градиентный фильтр Собеля
def sobel_grad(image):
	image_grad_x = cv.Sobel(image, cv.CV_32F, 1, 0)
	image_grad_y = cv.Sobel(image, cv.CV_32F, 0, 1)
	image_grad_x = cv.convertScaleAbs(image_grad_x)
	image_grad_y = cv.convertScaleAbs(image_grad_y)

	image_grad = cv.addWeighted(image_grad_x, 0.5, image_grad_y, 0.5, 0)
	#cv.imshow("sobel", image_grad)
	return image_grad

# Градиентный фильтр Шара
def scharr_grad(image):
	image_grad_x = cv.Scharr(image, cv.CV_32F, 1, 0)
	image_grad_y = cv.Scharr(image, cv.CV_32F, 0, 1)
	image_grad_x = cv.convertScaleAbs(image_grad_x)
	image_grad_y = cv.convertScaleAbs(image_grad_y)

	image_grad = cv.addWeighted(image_grad_x, 0.5, image_grad_y, 0.5, 0)
	#cv.imshow("scharr", image_grad)
	return image_grad

# возвращает изображение после применения градиентного фильтра
def get_gradient_image(image, gradient_filter):
	if gradient_filter == "sobel":
		gradient_image = sobel_grad(image)
	elif gradient_filter == "scharr":
		gradient_image = scharr_grad(image)
	else:
		gradient_image = laplace_grad(image)

	return gradient_image

# возвращает матрицу изображения - сумму r, g, b компонентов цветов
def get_sum_color_matrix(image, n, m, percent):
	sum_colors = np.zeros((n, m), dtype=int)
	for i in range(n):
		for j in range(m):
			value = int(image[i][j][0]) + int(image[i][j][1]) + int(image[i][j][2])
			
			if value <= percent / 100 * max_sum_of_colors:
				sum_colors[i][j] = 0
			else:
				sum_colors[i][j] = value

	return sum_colors

# Поиск максимальной прямоугольной области с градиентной закраской
def find_max_gradient_rectangle(image, percent):
	# прикрутим алгоритм поиска наибольшей нулевой подматрицы
	# обнуляем черные цвета, ищем по матрице сумм цветов

	n, m = image.shape[0], image.shape[1]

	max_rectangle_square = 0 # максимальная площадь прямоугольника
	max_rectangle = tuple()  # найденный прямоугольник

	sum_colors = get_sum_color_matrix(image, n, m, percent)

	nearest_not_zero_top = -np.ones(m, dtype=int)   # ближайшие не нули сверху
	nearest_not_zero_left = np.zeros(m, dtype=int)  # ближайшие не нули слева
	nearest_not_zero_right = np.zeros(m, dtype=int) # ближайшие не нули справа

	# алгоритм нахождения наибольшей нулевой подматрицы в матрице
	for i in range(n):
		# поиск ближайшего не нуля сверху
		for j in range(m):
			if sum_colors[i][j] >= 1:
				nearest_not_zero_top[j] = i

		# поиск ближайшего не нуля слева
		stack = list()
		for j in range(m):
			while len(stack) > 0 and nearest_not_zero_top[stack[-1]] <= nearest_not_zero_top[j]:
				stack.pop()
			if len(stack) != 0:
				nearest_not_zero_left[j] = stack[-1]
			else:
				nearest_not_zero_left[j] = -1
			stack.append(j)

		# поиск ближайшего не нуля справа
		stack = list()
		for j in range(m - 1, -1, -1):
			while len(stack) > 0 and nearest_not_zero_top[stack[-1]] <= nearest_not_zero_top[j]:
				stack.pop()
			if len(stack) != 0:
				nearest_not_zero_right[j] = stack[-1]
			else:
				nearest_not_zero_right[j] = m
			stack.append(j)

		# ищем и сохраняем наибольшие прямоугольник
		for j in range(m):
			current_square = (i - nearest_not_zero_top[j]) * \
							 (nearest_not_zero_right[j] - nearest_not_zero_left[j] - 1)
			rectangle = (nearest_not_zero_top[j] + 1, i, \
						 nearest_not_zero_left[j] + 1, nearest_not_zero_right[j] -1)

			if current_square > max_rectangle_square:
				max_rectangle = rectangle
				max_rectangle_square = current_square

	return max_rectangle

# найти n наибольших прямоугольников, являющихся градиентами, не пересекающихся
def find_n_greatest_gradient_rectangles(n, image, percent):
	rectangles = list()

	for i in range(n):
		rectangle = find_max_gradient_rectangle(image, percent)
		rectangles.append(rectangle)

		for j in range(rectangle[0], rectangle[1]):
			for k in range(rectangle[2], rectangle[3]):
				image[j][k][0] = image[j][k][1] = image[j][k][2] = 255

	return rectangles

# сохраняет изображение с выделенной областью с максимальным прямоугольником, 
# являющимся прямоугольникос
def save_image_with_rectangles(image, rectangles, directory):
	for rectangle in rectangles:
		image = cv.rectangle(image, (rectangle[2], rectangle[0]), (rectangle[3], rectangle[1]),
						 	 color=[0, 0, 255], thickness=3)
		cv.imwrite(str(directory/"result.png"), image)


if __name__ == "__main__":
# сохраним путь до файла с изображением
	# проверим, что как аргумент нам что-то передали
	if len(sys.argv) < 2:
		print("File name does not specified")
		exit()

	current_directory = pathlib.Path.cwd()
	image_path = current_directory/sys.argv[1]

	if not image_path.exists():
		print("File does not exists")
		exit()

# сохраним конфиг
	with open(current_directory/"data"/"config.json", "r") as config_file:
		config = json.load(config_file)

# откроем изображение
	image = cv.imread(str(image_path))
# сохраним изображение с применением градиентного фильтра
	gradient_image = get_gradient_image(image, config["gradient filter"])
	
# найдем n наибольших прямоугольников, являющихся градиентами (не пересекающихся)
	rectangles = find_n_greatest_gradient_rectangles(config["n"], gradient_image, config["percent"])
	print(rectangles)

# сохраним для наглядности в data картинку, где выделены найденные прямоугольники
	save_image_with_rectangles(image, rectangles, current_directory/"data")
