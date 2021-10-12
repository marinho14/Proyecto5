#Codigo realizado por Sebastian Marinho y Daniel Barandica, para la materia de Procesamiento de imagenes y video

# Se definen las librerias necesarias
import numpy as np
import cv2
import os


## Se definen algunas variables
points = [] ## Los puntos seleccionados por el usuario en las imagenes
H_list = [] ## La lista donde se guardaran las H
concat = [] ## Imagenes donde se guardara la concatenacion
flag = False


#Funcion recibir para pedir el path de las imagenes
def recibir():
    path = input("Ingrese la dirección de la carpeta donde se encuentras sus imagenes: ") #Se pide el path
    imagenes = [] #Lista para guardar imagenes
    cont = 1 #Contador
    while (True):
        try:
            #Se guardan las imagenes que estan en el path ingresado y que tengan como nombre image_# y formato JPEG
            image_name = "image_" + str(cont) + ".jpeg"
            path_file = os.path.join(path, image_name)
            image = cv2.imread(path_file)
            image = cv2.resize(image, (900, 980))
            imagenes.append(image)
            cont += 1
        except:
            break
    N = len(imagenes) #Numero de imagenes leidas
    return imagenes, N


#Funcion click para guardar los puntos que son puestos por cada usuario
def click(event, x, y, flags, param):
    global flag
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        flag = True

#Funcion Homography que fue tomada y adaptada de codigo realizado por Julian Quiroga
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


#Funcion promedio_images para promediar imagenes
def promedio_imagenes(img_1, img_2):
    #Binarizacion de las dos imagenes de entrada
    _, Ibw_1 = cv2.threshold(img_1[..., 0], 1, 255, cv2.THRESH_BINARY)
    _, Ibw_2 = cv2.threshold(img_2[..., 0], 1, 255, cv2.THRESH_BINARY)

    #Operacion And entre las imagenes binarizada
    mask = cv2.bitwise_and(Ibw_1, Ibw_2)

    #Operacion And entre mask y la imagen de entrada original a color
    img_1_l = cv2.bitwise_and(img_1, cv2.merge((mask, mask, mask)))
    img_2_l = cv2.bitwise_and(img_2, cv2.merge((mask, mask, mask)))


    #Conversion a uint32
    img_2_l = np.uint32(img_2_l)
    img_1_l = np.uint32(img_1_l)

    #Suma de ambas imagenes y division entera sobre 2
    img = np.uint8((img_2_l + img_1_l) // 2)

    #Mascara negada
    n_mask = cv2.bitwise_not(mask)

    #Operacion And entre n_mask y la imagen de entrada original a color
    img_1 = cv2.bitwise_and(img_1, cv2.merge((n_mask, n_mask, n_mask)))
    img_2 = cv2.bitwise_and(img_2, cv2.merge((n_mask, n_mask, n_mask)))

    #Operacion or entre las nuevas mascaras e img
    img = cv2.bitwise_or(img, img_1)
    img = cv2.bitwise_or(img, img_2)
    return img #Imagen promediada


#Funcion main
if __name__ == '__main__':
    #Se le pide al usuario ingresar el path de las imagenes y el numero de la imagen de referencia
    imagenes, N = recibir()
    print("El numero de imagenes recibidas es" + " " + str(N))
    ref = input("Escoja el numero de imagen de referencia: ")
    assert int(ref) <= N

    #Se realiza la homografia a cada una de las imagenes
    for i in range(N-1):
        a = Homography(imagenes[i], imagenes[(i + 1) % len(imagenes)])
        H_list.append(a)


    referencia = int(ref)-1 # Indice de imagen de referencia
    factor = 10 #Factor de escalado de la imagen
    des = 2200 #Factor de desplazamiento de la imagen

    h_traslacion = np.array([[1, 0, des], [0, 1, des], [0, 0, 1]], np.float64) #Matriz de desplazamiento

    img_transform = [] #Se guardan las imagenes de las diferentes perspectivas respecto a la referencia

    #Union de las homografias para creacion de imagen panoramica
    for i in range(N):
        h = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64) #Matriz identidad
        if i > referencia: #Se evalua para tomar las imagenes de la derecha de la referencia
            for cont, j in enumerate(H_list[referencia:i]):
                h = j @ h
            h = np.linalg.inv(h) #Al estar en la derecha se debe realizar la inversa
        elif i < referencia: #Se evalua para tomar las imagenes de la izquierda de la referencia
            for j in (H_list[i:referencia]):
                h = h @ j


        if i != referencia: #Se evalua que la imagen no sea la de referencia
            #Se proyecta las imagenes de entrada a la perspectiva de la referencia transalada
            img_warp = cv2.warpPerspective(imagenes[i], h_traslacion @ h,
                                           (imagenes[0].shape[1] * (factor), imagenes[0].shape[0] * (factor)))

        else:
            #Se traslada la imagen de referencia
            img_warp = cv2.warpPerspective(imagenes[i], h_traslacion,
                                           (imagenes[0].shape[1] * (factor), imagenes[0].shape[0] * (factor)))

        img_transform.append(img_warp)#Se guardan las imagenes obtenidas de warp

    prom = np.zeros_like(img_transform[i]) #Se crea una matriz de ceros
    for idx, img in enumerate(img_transform):
        prom = promedio_imagenes(prom, img)  #Promedio entre las imagenes obtenidas de la homografia

    cv2.imwrite("Imagen_panoramica.png", prom) #Se muestra la imagen resultante en pantalla
    cv2.waitKey(0)