# MC2016PRJ

Convolution 커널만 열심히 최적화 한 프로젝트.

# 미래의 나를 위한 꿀팁

## CPU

1. work-group = thread, work-item = 순차실행
2. Intel OpenCL guide를 보면 work-item들을 SIMD로 묶어 준다는데 안해주더라. AMD 플랫폼에서 Intel CPU를 써서 그런듯. 그냥 work-group 크기를 1로 하고 ```CL_DEVICE_PREFERRED_VECTOR_WIDTH_XXX```를 확인해서 맞춰주는 게 맘이 편하다. 여기선 float 기준 8이길래 float8 사용.

## GPU

1. 일단 이론상 최대 FLOPS를 확인하자. 성능이 그만큼 안나온다? 그러면 메모리 병목... GPU 연산속도는 메모리보다 무척 빠르다.
2. 행렬곱에서 메모리 사용량 줄이려면 blocking이 답. (막다 할때 block말고 구역 할때 block) n칸 단위 블록킹은 메모리 사용을 대충 n배 줄여준다.
3. 메모리가 global - local - register 이렇게 있으므로 두 수준에서 블록킹을 해줘야 한다. 속도가 몇배 차이나는지 확인하고 알맞게 해주자.

## 개선할 점 및 의문점

1. unroll을 하면 레지스터 사용량이 늘어나면서 오히려 느려진다. CUDA에서는 안그랬는데... 추측으로는 unroll 시에 생기는 상수들을 constant 메모리가 아니라 레지스터에 넣어서 쓰는 것 같다. 어셈 까서 확인 필요.
2. 나누기, 나머지 연산을 곱셈, 시프트 연산으로 대체 했는데 진짜 더 빠른지는 확인 안함. 확인 필요.
3. FC 레이어도 사실 행렬곱 커널 쓰면 더 빠르다. 귀찮아서 안 고침.
4. 사실 메모리 속도 고려하면 global에서는 128 단위 블록킹, local에서는 8 단위 블록킹을 해주어야 맞다. 근데 레이어 중에 크기가 64인 녀석이 있어서 귀찮아서 64, 4 단위 블록킹으로 함. 그래서 이론상 최대 성능의 1/4 정도만 나온다. 블록킹 크기별로 커널을 따로 만들어서 돌리면 최소 2배는 더 빨라질 것으로 예상.

<br>

--------
*MC2016PRJ* is primarily distributed under the terms of both the [MIT license]
and the [Apache License (Version 2.0)]. See [COPYRIGHT] for details.

[MIT license]: LICENSE-MIT
[Apache License (Version 2.0)]: LICENSE-APACHE
[COPYRIGHT]: COPYRIGHT
