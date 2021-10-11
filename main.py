# Se definen las librerias necesarias
import numpy as np
import cv2
import os


## Se definen algunas variables
points = [] ## Los puntos
H_list = [] ## La lista donde se guardaran las H
concat = []
flag = False


def recibir():
    path = input("Ingrese la dirección de la carpeta donde se encuentras sus imagenes: ")
    imagenes = []
    cont = 1
    while (True):
        try:
            image_name = "image_" + str(cont) + ".jpg"
            path_file = os.path.join(path, image_name)
            image = cv2.imread(path_file)
            image = cv2.resize(image, (900, 980))
            imagenes.append(image)
            cont += 1
        except:
            break
    N = len(imagenes)
    return imagenes, N


def click(event, x, y, flags, param):
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        flag = True


def Homography(image, image_2):
    global points, flag
    image_concat = cv2.hconcat([image, image_2])
    image_draw = image_concat.copy()

    points1 = []
    points2 = []

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", click)

    state = True  # Rojo
    while True:
        cv2.imshow("Image", image_draw)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("x"):
            break
        if flag:
            flag = False

            if (state):
                if (len(points2) > 0):
                    points2.pop(-1)
                    state = not state
            else:
                if (len(points1) > 0):
                    points1.pop(-1)
                    state = not state
            image_draw = image_concat.copy()
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [0, 0, 255], -1) for punto in points1]
            [cv2.circle(image_draw, (punto[0] + image.shape[1], punto[1]), 3, [255, 0, 0], -1) for punto in points2]

        if len(points) > 0:
            if (state):
                state = False
                points1.append((points[0][0], points[0][1]))
                points = []
            else:
                state = True
                points2.append((points[0][0] - image.shape[1], points[0][1]))
                points = []
            image_draw = image_concat.copy()
            [cv2.circle(image_draw, (punto[0], punto[1]), 3, [0, 0, 255], -1) for punto in points1]
            [cv2.circle(image_draw, (punto[0] + image.shape[1], punto[1]), 3, [255, 0, 0], -1) for punto in points2]

    N = min(len(points1), len(points2))

    cv2.destroyAllWindows()
    assert N >= 4, 'At least four points are required'

    pts1 = np.array(points1[:N])
    pts2 = np.array(points2[:N])

    if False:
        H, _ = cv2.findHomography(pts1, pts2, method=0)
    else:
        H, _ = cv2.findHomography(pts1, pts2, method=cv2.RANSAC)
    return H


def promedio_imagenes(img_1, img_2):
    _, Ibw_1 = cv2.threshold(img_1[..., 0], 1, 255, cv2.THRESH_BINARY)
    _, Ibw_2 = cv2.threshold(img_2[..., 0], 1, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(Ibw_1, Ibw_2)

    img_1_l = cv2.bitwise_and(img_1, cv2.merge((mask, mask, mask)))
    img_2_l = cv2.bitwise_and(img_2, cv2.merge((mask, mask, mask)))

    # cv2.imshow("mascara_1",img_1_l)

    img_2_l = np.uint32(img_2_l)
    img_1_l = np.uint32(img_1_l)

    img = np.uint8((img_2_l + img_1_l) // 2)

    # cv2.imshow("mascara_2",img)

    n_mask = cv2.bitwise_not(mask)

    img_1 = cv2.bitwise_and(img_1, cv2.merge((n_mask, n_mask, n_mask)))
    img_2 = cv2.bitwise_and(img_2, cv2.merge((n_mask, n_mask, n_mask)))

    # cv2.imshow("mascara_2",img_1)

    img = cv2.bitwise_or(img, img_1)
    img = cv2.bitwise_or(img, img_2)
    return img


if __name__ == '__main__':
    imagenes, N = recibir()
    print("El numero de imagenes recibidas es" + " " + str(N))
    ref = input("Escoja el numero de imagen de referencia: ")
    assert int(ref) <= N
    for i in range(N-1):  ## N-1
        a = Homography(imagenes[i], imagenes[(i + 1) % len(imagenes)])
        H_list.append(a)

    referencia = int(ref)-1
    factor = 10

    I = np.identity(H_list[-1].shape[0])
    des = 2500
    h_traslacion = np.array([[1, 0, des], [0, 1, des], [0, 0, 1]], np.float64)

    img_transform = []
    img_recortada = []
    for i in range(N):
        h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
        print("Iter es:", i)
        if i > referencia:
            for cont, j in enumerate(H_list[referencia:i]):
                h = j @ h
            h = np.linalg.inv(h)
        elif i < referencia:
            for j in (H_list[i:referencia]):
                h = h @ j
        if i != referencia:
            img_wrap = cv2.warpPerspective(imagenes[i], h_traslacion @ h,
                                           (imagenes[0].shape[1] * (factor), imagenes[0].shape[0] * (factor)))

        else:
            img_wrap = cv2.warpPerspective(imagenes[i], h_traslacion,
                                           (imagenes[0].shape[1] * (factor), imagenes[0].shape[0] * (factor)))

        img_transform.append(img_wrap)

    prom = np.zeros_like(img_transform[i])
    for idx, img in enumerate(img_transform):
        prom = promedio_imagenes(prom, img)

    cv2.imwrite("res3.png", prom)
    cv2.waitKey(0)