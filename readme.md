### ratent space experiments

- input : embedding vector using CBOW Embeeding from korean lyrics datasets
- use model : auto encoder, kmeans, t-sne
- result insight images : latent_space vector, kmeans, t-sne clustering

<h3>Important</h3>
<ol>
    <li>
        PCA의 차원축소 양상과 AutoEncoder의 차원 축소의 양상이 거의 동일하였다.
    </li>
    <li>
        더 성능이 좋은 DEC 모델의 경우도 위와 같은 흐름이지만 더욱 최적화된 군집을 보여줄 수 있을 것으로 생각됨.
    </li>
    <li>
        발라드 가사 중심으로 군집한 결과 최적의 군집이 13개 정도 였지만 희소 데이터를 고려하면 군집은 9개 정도로 축약될 수 있음
    </li>
</ol>