from django.shortcuts import render, redirect
from .forms import ImageForm
from .models import Image
from bs4 import BeautifulSoup
import requests


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Create your views here.

def index(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        images = Image.objects.all()
        if len(images) == 1:
            delete_image = Image.objects.all()
            delete_image.delete()
            
            image = form.save(commit=False)
            image.save()
            images = Image.objects.all()
            image_path = images[0].image
            top_3, percent= run_inference_on_image(str(image_path))

            car_name_list, car_type_list, car_producer_list, car_image_list, car_detail_list = car_crawling(top_3)
    
            form = ImageForm()
            context = {
                'images':images,
                'percent':percent,
                'form':form,
                'car_1_name':car_name_list[0],
                'car_1_type':car_type_list[0],
                'car_1_producer':car_producer_list[0],
                'car_1_image':car_image_list[0],
                'car_1_detail':car_detail_list[0],
                'car_2_name':car_name_list[1],
                'car_2_type':car_type_list[1],
                'car_2_producer':car_producer_list[1],
                'car_2_image':car_image_list[1],
                'car_2_detail':car_detail_list[1],
                'car_3_name':car_name_list[2],
                'car_3_type':car_type_list[2],
                'car_3_producer':car_producer_list[2],
                'car_3_image':car_image_list[2],
                'car_3_detail':car_detail_list[2],
            }

            return render(request, 'result.html',context)
        else:
            if form.is_valid():
                image = form.save(commit=False)
                image.save()
                images = Image.objects.all()
                image_path = images[0].image
                top_3, percent= run_inference_on_image(str(image_path))
                car_name_list, car_type_list, car_producer_list, car_image_list, car_detail_list = car_crawling(top_3)
    
                form = ImageForm()
                context = {
                    'images':images,
                    'percent':percent,
                    'form':form,
                    'car_1_name':car_name_list[0],
                    'car_1_type':car_type_list[0],
                    'car_1_producer':car_producer_list[0],
                    'car_1_image':car_image_list[0],
                    'car_1_detail':car_detail_list[0],
                    'car_2_name':car_name_list[1],
                    'car_2_type':car_type_list[1],
                    'car_2_producer':car_producer_list[1],
                    'car_2_image':car_image_list[1],
                    'car_2_detail':car_detail_list[1],
                    'car_3_name':car_name_list[2],
                    'car_3_type':car_type_list[2],
                    'car_3_producer':car_producer_list[2],
                    'car_3_image':car_image_list[2],
                    'car_3_detail':car_detail_list[2],
                }

                return render(request, 'result.html',context)
    else:
        delete_image=Image.objects.all()
        delete_image.delete()
        form = ImageForm()
        context = {
            'form':form
        }
        return render(request, 'index.html', context)

def result(request):
    return render(request, 'result.html')

def run_inference_on_image(image_path):
    # 추론을 진행할 이미지 파일경로
    # image_path = './kia3.jpg'
    # image_path = './model_360.png'

    # 읽어들일 labels 파일 경로
    labels_txt_file_path = './output_labels.txt'
    answer = None

    # 만약 경로에 이미지 파일이 없을 경우 오류 로그를 출력합니다.
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('추론할 이미지 파일이 존재하지 않습니다. %s', image_path)
        return answer

    # 이미지 파일을 읽습니다.
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # 그래프를 생성합니다.
    graph_pb_file_path = './output_graph.pb'
    with tf.gfile.FastGFile(graph_pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # 세션을 열고 그래프를 실행합니다.
    with tf.Session() as sess:
        # 최종 소프트 맥스 출력 레이어를 지정합니다.
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        # 추론할 이미지를 인풋으로 넣고 추론 결과인 소프트 맥스 행렬을 리턴 받습니다.
        predictions = sess.run(softmax_tensor, feed_dict={
                               'DecodeJpeg/contents:0': image_data})
        # 불필요한 차원을 제거합니다.
        predictions = np.squeeze(predictions)
        # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)들의 인덱스를 가져옵니다.
        # e.g. [0 3 2 4 1]]
        top_k = predictions.argsort()[-5:][::-1]

        # output_labels.txt 파일로부터 정답 레이블들을 list 형태로 가져옵니다.
        f = open(labels_txt_file_path, 'r', encoding='utf-8')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        # 가장 높은 확률을 가진 인덱스들부터 추론 결과(Top-10)를 출력합니다.
        top_3 = []
        percent = []

        idx = 0
        for node_id in top_k:
            idx += 1
            label_name = labels[node_id]
            probability = predictions[node_id]
            top_3.append(label_name)
            percent.append(probability)
            if idx == 3:
                break
        

        # 가장 높은 확류을 가진 Top-1 추론 결과를 출력합니다.
        answer = labels[top_k[0]]
        probability = predictions[top_k[0]]
    
    return top_3, percent

def car_crawling(top_3):
    car_name_list = []
    car_producer_list = []
    car_type_list = []
    car_image_list = []
    car_detail_list = []
    for i in range(3):
        full_name = top_3[i]
    
        idx = 0
        check_list = []
        for j in full_name:
            if j == '_':
                check_list.append(idx)
            idx += 1
        car_name = full_name[check_list[0]+1:check_list[1]]
        car_year = full_name[check_list[1]+1:]
    
        if car_name == 'QM6' and car_year == '2017':
            car_year = '2016'
        if car_name == '투싼1.7D':
            car_name = '투싼'

        car_url = 'https://search.naver.com/search.naver?sm=tab_hty.top&where=nexearch&query='+car_year+car_name
        resp = requests.get(car_url)
        soup = BeautifulSoup(resp.content, 'html.parser')


        try:
            car_name_year = soup.select('#main_pack > div.content_search.section > div > div.contents03_sub > div > div.profile_wrap > dl > dt.name > a > strong')[0].text
            car_name_list.append(car_name_year)       

        except:
            car_name_list.append(car_name + car_year)
        try:
            car_detail = soup.select('#main_pack > div.content_search.section > div > div.contents03_sub > div > div.profile_wrap > dl > dt.name > a')[0].get('href')
            car_detail_list.append(car_detail)
            detail_url = car_detail
            resp = requests.get(detail_url)
            soup = BeautifulSoup(resp.content, 'html.parser')
            car_type = soup.select('#container > div.spot_end.new_end > div.info_group > span > a.weight')[0].text
            car_type_list.append(car_type)
            car_producer = soup.select('#container > div.spot_end.new_end > div.info_group > span > a.brand')[0].text
            car_producer_list.append(car_producer)
            car_image = soup.select('#carMainImgArea > div.main_img > img')[0].get('src')
            car_image_list.append(car_image)

        except:
            car_detail_list.append('확인불가로 네이버검색 요망')
            car_type_list.append('확인불가로 네이버검색 요망')
            car_producer_list.append('확인불가로 네이버검색 요망')
            car_image_list.append('확인불가로 네이버검색 요망')
    return car_name_list, car_type_list, car_producer_list, car_image_list, car_detail_list
