��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
p
AudioSpectrogram	
input
spectrogram"
window_sizeint"
strideint"
magnitude_squaredbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
�
Mfcc
spectrogram
sample_rate

output"%
upper_frequency_limitfloat%  zE"%
lower_frequency_limitfloat%  �A"#
filterbank_channel_countint(" 
dct_coefficient_countint

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58��
�
svdf_5/stream_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_namesvdf_5/stream_5/bias
z
(svdf_5/stream_5/bias/Read/ReadVariableOpReadVariableOpsvdf_5/stream_5/bias*
_output_shapes	
:�*
dtype0
�
 svdf_5/stream_5/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�*1
shared_name" svdf_5/stream_5/depthwise_kernel
�
4svdf_5/stream_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp svdf_5/stream_5/depthwise_kernel*'
_output_shapes
:
�*
dtype0
�
svdf_5/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*'
shared_namesvdf_5/dense_10/kernel
�
*svdf_5/dense_10/kernel/Read/ReadVariableOpReadVariableOpsvdf_5/dense_10/kernel*
_output_shapes
:	@�*
dtype0
~
svdf_4/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesvdf_4/dense_9/bias
w
'svdf_4/dense_9/bias/Read/ReadVariableOpReadVariableOpsvdf_4/dense_9/bias*
_output_shapes
:@*
dtype0
�
svdf_4/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_namesvdf_4/dense_9/kernel

)svdf_4/dense_9/kernel/Read/ReadVariableOpReadVariableOpsvdf_4/dense_9/kernel*
_output_shapes

:@@*
dtype0
�
svdf_4/stream_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namesvdf_4/stream_4/bias
y
(svdf_4/stream_4/bias/Read/ReadVariableOpReadVariableOpsvdf_4/stream_4/bias*
_output_shapes
:@*
dtype0
�
 svdf_4/stream_4/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*1
shared_name" svdf_4/stream_4/depthwise_kernel
�
4svdf_4/stream_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp svdf_4/stream_4/depthwise_kernel*&
_output_shapes
:
@*
dtype0
�
svdf_4/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_namesvdf_4/dense_8/kernel

)svdf_4/dense_8/kernel/Read/ReadVariableOpReadVariableOpsvdf_4/dense_8/kernel*
_output_shapes

:@@*
dtype0
~
svdf_3/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesvdf_3/dense_7/bias
w
'svdf_3/dense_7/bias/Read/ReadVariableOpReadVariableOpsvdf_3/dense_7/bias*
_output_shapes
:@*
dtype0
�
svdf_3/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*&
shared_namesvdf_3/dense_7/kernel

)svdf_3/dense_7/kernel/Read/ReadVariableOpReadVariableOpsvdf_3/dense_7/kernel*
_output_shapes

: @*
dtype0
�
svdf_3/stream_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namesvdf_3/stream_3/bias
y
(svdf_3/stream_3/bias/Read/ReadVariableOpReadVariableOpsvdf_3/stream_3/bias*
_output_shapes
: *
dtype0
�
 svdf_3/stream_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *1
shared_name" svdf_3/stream_3/depthwise_kernel
�
4svdf_3/stream_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp svdf_3/stream_3/depthwise_kernel*&
_output_shapes
:
 *
dtype0
�
svdf_3/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *&
shared_namesvdf_3/dense_6/kernel

)svdf_3/dense_6/kernel/Read/ReadVariableOpReadVariableOpsvdf_3/dense_6/kernel*
_output_shapes

:@ *
dtype0
~
svdf_2/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_namesvdf_2/dense_5/bias
w
'svdf_2/dense_5/bias/Read/ReadVariableOpReadVariableOpsvdf_2/dense_5/bias*
_output_shapes
:@*
dtype0
�
svdf_2/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: @*&
shared_namesvdf_2/dense_5/kernel

)svdf_2/dense_5/kernel/Read/ReadVariableOpReadVariableOpsvdf_2/dense_5/kernel*
_output_shapes

: @*
dtype0
�
svdf_2/stream_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namesvdf_2/stream_2/bias
y
(svdf_2/stream_2/bias/Read/ReadVariableOpReadVariableOpsvdf_2/stream_2/bias*
_output_shapes
: *
dtype0
�
 svdf_2/stream_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *1
shared_name" svdf_2/stream_2/depthwise_kernel
�
4svdf_2/stream_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp svdf_2/stream_2/depthwise_kernel*&
_output_shapes
:
 *
dtype0
�
svdf_2/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *&
shared_namesvdf_2/dense_4/kernel

)svdf_2/dense_4/kernel/Read/ReadVariableOpReadVariableOpsvdf_2/dense_4/kernel*
_output_shapes

:( *
dtype0
~
svdf_1/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_namesvdf_1/dense_3/bias
w
'svdf_1/dense_3/bias/Read/ReadVariableOpReadVariableOpsvdf_1/dense_3/bias*
_output_shapes
:(*
dtype0
�
svdf_1/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: (*&
shared_namesvdf_1/dense_3/kernel

)svdf_1/dense_3/kernel/Read/ReadVariableOpReadVariableOpsvdf_1/dense_3/kernel*
_output_shapes

: (*
dtype0
�
svdf_1/stream_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namesvdf_1/stream_1/bias
y
(svdf_1/stream_1/bias/Read/ReadVariableOpReadVariableOpsvdf_1/stream_1/bias*
_output_shapes
: *
dtype0
�
 svdf_1/stream_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *1
shared_name" svdf_1/stream_1/depthwise_kernel
�
4svdf_1/stream_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp svdf_1/stream_1/depthwise_kernel*&
_output_shapes
:
 *
dtype0
�
svdf_1/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:( *&
shared_namesvdf_1/dense_2/kernel

)svdf_1/dense_2/kernel/Read/ReadVariableOpReadVariableOpsvdf_1/dense_2/kernel*
_output_shapes

:( *
dtype0
~
svdf_0/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*$
shared_namesvdf_0/dense_1/bias
w
'svdf_0/dense_1/bias/Read/ReadVariableOpReadVariableOpsvdf_0/dense_1/bias*
_output_shapes
:(*
dtype0
�
svdf_0/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*&
shared_namesvdf_0/dense_1/kernel

)svdf_0/dense_1/kernel/Read/ReadVariableOpReadVariableOpsvdf_0/dense_1/kernel*
_output_shapes

:(*
dtype0
|
svdf_0/stream/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namesvdf_0/stream/bias
u
&svdf_0/stream/bias/Read/ReadVariableOpReadVariableOpsvdf_0/stream/bias*
_output_shapes
:*
dtype0
�
svdf_0/stream/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name svdf_0/stream/depthwise_kernel
�
2svdf_0/stream/depthwise_kernel/Read/ReadVariableOpReadVariableOpsvdf_0/stream/depthwise_kernel*&
_output_shapes
:*
dtype0
�
svdf_0/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(*$
shared_namesvdf_0/dense/kernel
{
'svdf_0/dense/kernel/Read/ReadVariableOpReadVariableOpsvdf_0/dense/kernel*
_output_shapes

:(*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	�*
dtype0
j
serving_default_input_1Placeholder*
_output_shapes
:	�}*
dtype0*
shape:	�}
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1svdf_0/dense/kernelsvdf_0/stream/depthwise_kernelsvdf_0/stream/biassvdf_0/dense_1/kernelsvdf_0/dense_1/biassvdf_1/dense_2/kernel svdf_1/stream_1/depthwise_kernelsvdf_1/stream_1/biassvdf_1/dense_3/kernelsvdf_1/dense_3/biassvdf_2/dense_4/kernel svdf_2/stream_2/depthwise_kernelsvdf_2/stream_2/biassvdf_2/dense_5/kernelsvdf_2/dense_5/biassvdf_3/dense_6/kernel svdf_3/stream_3/depthwise_kernelsvdf_3/stream_3/biassvdf_3/dense_7/kernelsvdf_3/dense_7/biassvdf_4/dense_8/kernel svdf_4/stream_4/depthwise_kernelsvdf_4/stream_4/biassvdf_4/dense_9/kernelsvdf_4/dense_9/biassvdf_5/dense_10/kernel svdf_5/stream_5/depthwise_kernelsvdf_5/stream_5/biasdense_11/kerneldense_11/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3795

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
valueڈBֈ BΈ
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

params

rand_shift
rand_stretch_squeeze

data_frame
	add_noise
preemphasis
 	windowing
!mag_rdft_mel
"log_max
#dct
$
normalizer
%spec_augment
&spec_cutout* 
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-dropout1

.dense1
/
depth_cnn1

0dense2
1
batch_norm*
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8dropout1

9dense1
:
depth_cnn1

;dense2
<
batch_norm*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cdropout1

Ddense1
E
depth_cnn1

Fdense2
G
batch_norm*
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Ndropout1

Odense1
P
depth_cnn1

Qdense2
R
batch_norm*
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Ydropout1

Zdense1
[
depth_cnn1

\dense2
]
batch_norm*
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
ddropout1

edense1
f
depth_cnn1

gdense2
h
batch_norm*
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
ocell
pstate_shape* 
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_random_generator* 
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
~28
29*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
~28
29*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
&
�	keras_api
�padding_layer* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 

�	keras_api* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�mean
�stddev* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
,
�0
�1
�2
�3
�4*
,
�0
�1
�2
�3
�4*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
,
�0
�1
�2
�3
�4*
,
�0
�1
�2
�3
�4*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
,
�0
�1
�2
�3
�4*
,
�0
�1
�2
�3
�4*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
,
�0
�1
�2
�3
�4*
,
�0
�1
�2
�3
�4*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
,
�0
�1
�2
�3
�4*
,
�0
�1
�2
�3
�4*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 

�0
�1
�2*

�0
�1
�2*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

~0
1*

~0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEsvdf_0/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEsvdf_0/stream/depthwise_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEsvdf_0/stream/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsvdf_0/dense_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEsvdf_0/dense_1/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsvdf_1/dense_2/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUE svdf_1/stream_1/depthwise_kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEsvdf_1/stream_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsvdf_1/dense_3/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEsvdf_1/dense_3/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEsvdf_2/dense_4/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE svdf_2/stream_2/depthwise_kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsvdf_2/stream_2/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEsvdf_2/dense_5/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEsvdf_2/dense_5/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEsvdf_3/dense_6/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE svdf_3/stream_3/depthwise_kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsvdf_3/stream_3/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEsvdf_3/dense_7/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEsvdf_3/dense_7/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEsvdf_4/dense_8/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE svdf_4/stream_4/depthwise_kernel'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsvdf_4/stream_4/bias'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEsvdf_4/dense_9/kernel'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEsvdf_4/dense_9/bias'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEsvdf_5/dense_10/kernel'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE svdf_5/stream_5/depthwise_kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEsvdf_5/stream_5/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
X
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�	keras_api* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
'
-0
.1
/2
03
14*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
'
80
91
:2
;3
<4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
'
C0
D1
E2
F3
G4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
'
N0
O1
P2
Q3
R4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
'
Y0
Z1
[2
\3
]4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
'
d0
e1
f2
g3
h4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
	
o0* 
* 
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0*
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp'svdf_0/dense/kernel/Read/ReadVariableOp2svdf_0/stream/depthwise_kernel/Read/ReadVariableOp&svdf_0/stream/bias/Read/ReadVariableOp)svdf_0/dense_1/kernel/Read/ReadVariableOp'svdf_0/dense_1/bias/Read/ReadVariableOp)svdf_1/dense_2/kernel/Read/ReadVariableOp4svdf_1/stream_1/depthwise_kernel/Read/ReadVariableOp(svdf_1/stream_1/bias/Read/ReadVariableOp)svdf_1/dense_3/kernel/Read/ReadVariableOp'svdf_1/dense_3/bias/Read/ReadVariableOp)svdf_2/dense_4/kernel/Read/ReadVariableOp4svdf_2/stream_2/depthwise_kernel/Read/ReadVariableOp(svdf_2/stream_2/bias/Read/ReadVariableOp)svdf_2/dense_5/kernel/Read/ReadVariableOp'svdf_2/dense_5/bias/Read/ReadVariableOp)svdf_3/dense_6/kernel/Read/ReadVariableOp4svdf_3/stream_3/depthwise_kernel/Read/ReadVariableOp(svdf_3/stream_3/bias/Read/ReadVariableOp)svdf_3/dense_7/kernel/Read/ReadVariableOp'svdf_3/dense_7/bias/Read/ReadVariableOp)svdf_4/dense_8/kernel/Read/ReadVariableOp4svdf_4/stream_4/depthwise_kernel/Read/ReadVariableOp(svdf_4/stream_4/bias/Read/ReadVariableOp)svdf_4/dense_9/kernel/Read/ReadVariableOp'svdf_4/dense_9/bias/Read/ReadVariableOp*svdf_5/dense_10/kernel/Read/ReadVariableOp4svdf_5/stream_5/depthwise_kernel/Read/ReadVariableOp(svdf_5/stream_5/bias/Read/ReadVariableOpConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_4974
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/biassvdf_0/dense/kernelsvdf_0/stream/depthwise_kernelsvdf_0/stream/biassvdf_0/dense_1/kernelsvdf_0/dense_1/biassvdf_1/dense_2/kernel svdf_1/stream_1/depthwise_kernelsvdf_1/stream_1/biassvdf_1/dense_3/kernelsvdf_1/dense_3/biassvdf_2/dense_4/kernel svdf_2/stream_2/depthwise_kernelsvdf_2/stream_2/biassvdf_2/dense_5/kernelsvdf_2/dense_5/biassvdf_3/dense_6/kernel svdf_3/stream_3/depthwise_kernelsvdf_3/stream_3/biassvdf_3/dense_7/kernelsvdf_3/dense_7/biassvdf_4/dense_8/kernel svdf_4/stream_4/depthwise_kernelsvdf_4/stream_4/biassvdf_4/dense_9/kernelsvdf_4/dense_9/biassvdf_5/dense_10/kernel svdf_5/stream_5/depthwise_kernelsvdf_5/stream_5/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_5074��
�
e
I__inference_speech_features_layer_call_and_return_conditional_losses_3404

inputs
identity_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       a
	transpose	Transposeinputstranspose/perm:output:0*
T0*
_output_shapes
:	�}�
AudioSpectrogramAudioSpectrogramtranspose:y:0*#
_output_shapes
:1�*
magnitude_squared(*
stride�*
window_size�S
Mfcc/sample_rateConst*
_output_shapes
: *
dtype0*
value
B :�}�
MfccMfccAudioSpectrogram:spectrogram:0Mfcc/sample_rate:output:0*"
_output_shapes
:1(*
dct_coefficient_count(*
filterbank_channel_countP*
upper_frequency_limit% ��E�
normalizer/sub/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�                                                                                                                                                                l
normalizer/subSubMfcc:output:0normalizer/sub/y:output:0*
T0*"
_output_shapes
:1(�
normalizer/truediv/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?}
normalizer/truedivRealDivnormalizer/sub:z:0normalizer/truediv/y:output:0*
T0*"
_output_shapes
:1(Y
IdentityIdentitynormalizer/truediv:z:0*
T0*"
_output_shapes
:1("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	�}:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_2555

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	�S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	�:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_2636

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	�S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	�:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�9
�
?__inference_model_layer_call_and_return_conditional_losses_2572

inputs,
svdf_0_svdf_0_dense_kernel:(?
%svdf_0_svdf_0_stream_depthwise_kernel:'
svdf_0_svdf_0_stream_bias:.
svdf_0_svdf_0_dense_1_kernel:((
svdf_0_svdf_0_dense_1_bias:(.
svdf_1_svdf_1_dense_2_kernel:( A
'svdf_1_svdf_1_stream_1_depthwise_kernel:
 )
svdf_1_svdf_1_stream_1_bias: .
svdf_1_svdf_1_dense_3_kernel: ((
svdf_1_svdf_1_dense_3_bias:(.
svdf_2_svdf_2_dense_4_kernel:( A
'svdf_2_svdf_2_stream_2_depthwise_kernel:
 )
svdf_2_svdf_2_stream_2_bias: .
svdf_2_svdf_2_dense_5_kernel: @(
svdf_2_svdf_2_dense_5_bias:@.
svdf_3_svdf_3_dense_6_kernel:@ A
'svdf_3_svdf_3_stream_3_depthwise_kernel:
 )
svdf_3_svdf_3_stream_3_bias: .
svdf_3_svdf_3_dense_7_kernel: @(
svdf_3_svdf_3_dense_7_bias:@.
svdf_4_svdf_4_dense_8_kernel:@@A
'svdf_4_svdf_4_stream_4_depthwise_kernel:
@)
svdf_4_svdf_4_stream_4_bias:@.
svdf_4_svdf_4_dense_9_kernel:@@(
svdf_4_svdf_4_dense_9_bias:@0
svdf_5_svdf_5_dense_10_kernel:	@�B
'svdf_5_svdf_5_stream_5_depthwise_kernel:
�*
svdf_5_svdf_5_stream_5_bias:	�+
dense_11_dense_11_kernel:	�$
dense_11_dense_11_bias:
identity�� dense_11/StatefulPartitionedCall�svdf_0/StatefulPartitionedCall�svdf_1/StatefulPartitionedCall�svdf_2/StatefulPartitionedCall�svdf_3/StatefulPartitionedCall�svdf_4/StatefulPartitionedCall�svdf_5/StatefulPartitionedCall�
speech_features/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_speech_features_layer_call_and_return_conditional_losses_2300�
svdf_0/StatefulPartitionedCallStatefulPartitionedCall(speech_features/PartitionedCall:output:0svdf_0_svdf_0_dense_kernel%svdf_0_svdf_0_stream_depthwise_kernelsvdf_0_svdf_0_stream_biassvdf_0_svdf_0_dense_1_kernelsvdf_0_svdf_0_dense_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:.(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_0_layer_call_and_return_conditional_losses_2337�
svdf_1/StatefulPartitionedCallStatefulPartitionedCall'svdf_0/StatefulPartitionedCall:output:0svdf_1_svdf_1_dense_2_kernel'svdf_1_svdf_1_stream_1_depthwise_kernelsvdf_1_svdf_1_stream_1_biassvdf_1_svdf_1_dense_3_kernelsvdf_1_svdf_1_dense_3_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:%(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_1_layer_call_and_return_conditional_losses_2379�
svdf_2/StatefulPartitionedCallStatefulPartitionedCall'svdf_1/StatefulPartitionedCall:output:0svdf_2_svdf_2_dense_4_kernel'svdf_2_svdf_2_stream_2_depthwise_kernelsvdf_2_svdf_2_stream_2_biassvdf_2_svdf_2_dense_5_kernelsvdf_2_svdf_2_dense_5_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_2_layer_call_and_return_conditional_losses_2421�
svdf_3/StatefulPartitionedCallStatefulPartitionedCall'svdf_2/StatefulPartitionedCall:output:0svdf_3_svdf_3_dense_6_kernel'svdf_3_svdf_3_stream_3_depthwise_kernelsvdf_3_svdf_3_stream_3_biassvdf_3_svdf_3_dense_7_kernelsvdf_3_svdf_3_dense_7_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_3_layer_call_and_return_conditional_losses_2463�
svdf_4/StatefulPartitionedCallStatefulPartitionedCall'svdf_3/StatefulPartitionedCall:output:0svdf_4_svdf_4_dense_8_kernel'svdf_4_svdf_4_stream_4_depthwise_kernelsvdf_4_svdf_4_stream_4_biassvdf_4_svdf_4_dense_9_kernelsvdf_4_svdf_4_dense_9_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:
@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_4_layer_call_and_return_conditional_losses_2505�
svdf_5/StatefulPartitionedCallStatefulPartitionedCall'svdf_4/StatefulPartitionedCall:output:0svdf_5_svdf_5_dense_10_kernel'svdf_5_svdf_5_stream_5_depthwise_kernelsvdf_5_svdf_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:�*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_5_layer_call_and_return_conditional_losses_2537�
stream_6/PartitionedCallPartitionedCall'svdf_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_2548�
dropout/PartitionedCallPartitionedCall!stream_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2555�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_2567o
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp!^dense_11/StatefulPartitionedCall^svdf_0/StatefulPartitionedCall^svdf_1/StatefulPartitionedCall^svdf_2/StatefulPartitionedCall^svdf_3/StatefulPartitionedCall^svdf_4/StatefulPartitionedCall^svdf_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
svdf_0/StatefulPartitionedCallsvdf_0/StatefulPartitionedCall2@
svdf_1/StatefulPartitionedCallsvdf_1/StatefulPartitionedCall2@
svdf_2/StatefulPartitionedCallsvdf_2/StatefulPartitionedCall2@
svdf_3/StatefulPartitionedCallsvdf_3/StatefulPartitionedCall2@
svdf_4/StatefulPartitionedCallsvdf_4/StatefulPartitionedCall2@
svdf_5/StatefulPartitionedCallsvdf_5/StatefulPartitionedCall:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�
� 
?__inference_model_layer_call_and_return_conditional_losses_4261

inputsK
9svdf_0_dense_tensordot_readvariableop_svdf_0_dense_kernel:(p
Vsvdf_0_stream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel:V
Hsvdf_0_stream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias:O
=svdf_0_dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel:(G
9svdf_0_dense_1_biasadd_readvariableop_svdf_0_dense_1_bias:(O
=svdf_1_dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel:( v
\svdf_1_stream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel:
 \
Nsvdf_1_stream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias: O
=svdf_1_dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel: (G
9svdf_1_dense_3_biasadd_readvariableop_svdf_1_dense_3_bias:(O
=svdf_2_dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel:( v
\svdf_2_stream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel:
 \
Nsvdf_2_stream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias: O
=svdf_2_dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel: @G
9svdf_2_dense_5_biasadd_readvariableop_svdf_2_dense_5_bias:@O
=svdf_3_dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel:@ v
\svdf_3_stream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel:
 \
Nsvdf_3_stream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias: O
=svdf_3_dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel: @G
9svdf_3_dense_7_biasadd_readvariableop_svdf_3_dense_7_bias:@O
=svdf_4_dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel:@@v
\svdf_4_stream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel:
@\
Nsvdf_4_stream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias:@O
=svdf_4_dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel:@@G
9svdf_4_dense_9_biasadd_readvariableop_svdf_4_dense_9_bias:@R
?svdf_5_dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel:	@�w
\svdf_5_stream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel:
�]
Nsvdf_5_stream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias:	�A
.dense_11_matmul_readvariableop_dense_11_kernel:	�;
-dense_11_biasadd_readvariableop_dense_11_bias:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�%svdf_0/dense/Tensordot/ReadVariableOp�%svdf_0/dense_1/BiasAdd/ReadVariableOp�'svdf_0/dense_1/Tensordot/ReadVariableOp�5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp�7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp�'svdf_1/dense_2/Tensordot/ReadVariableOp�%svdf_1/dense_3/BiasAdd/ReadVariableOp�'svdf_1/dense_3/Tensordot/ReadVariableOp�9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp�;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp�'svdf_2/dense_4/Tensordot/ReadVariableOp�%svdf_2/dense_5/BiasAdd/ReadVariableOp�'svdf_2/dense_5/Tensordot/ReadVariableOp�9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp�;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp�'svdf_3/dense_6/Tensordot/ReadVariableOp�%svdf_3/dense_7/BiasAdd/ReadVariableOp�'svdf_3/dense_7/Tensordot/ReadVariableOp�9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp�;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp�'svdf_4/dense_8/Tensordot/ReadVariableOp�%svdf_4/dense_9/BiasAdd/ReadVariableOp�'svdf_4/dense_9/Tensordot/ReadVariableOp�9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp�;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp�(svdf_5/dense_10/Tensordot/ReadVariableOp�9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp�;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpo
speech_features/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
speech_features/transpose	Transposeinputs'speech_features/transpose/perm:output:0*
T0*
_output_shapes
:	�}�
 speech_features/AudioSpectrogramAudioSpectrogramspeech_features/transpose:y:0*#
_output_shapes
:1�*
magnitude_squared(*
stride�*
window_size�c
 speech_features/Mfcc/sample_rateConst*
_output_shapes
: *
dtype0*
value
B :�}�
speech_features/MfccMfcc.speech_features/AudioSpectrogram:spectrogram:0)speech_features/Mfcc/sample_rate:output:0*"
_output_shapes
:1(*
dct_coefficient_count(*
filterbank_channel_countP*
upper_frequency_limit% ��E�
 speech_features/normalizer/sub/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�                                                                                                                                                                �
speech_features/normalizer/subSubspeech_features/Mfcc:output:0)speech_features/normalizer/sub/y:output:0*
T0*"
_output_shapes
:1(�
$speech_features/normalizer/truediv/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?�
"speech_features/normalizer/truedivRealDiv"speech_features/normalizer/sub:z:0-speech_features/normalizer/truediv/y:output:0*
T0*"
_output_shapes
:1(W
svdf_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_0/ExpandDims
ExpandDims&speech_features/normalizer/truediv:z:0svdf_0/ExpandDims/dim:output:0*
T0*&
_output_shapes
:1(�
%svdf_0/dense/Tensordot/ReadVariableOpReadVariableOp9svdf_0_dense_tensordot_readvariableop_svdf_0_dense_kernel*
_output_shapes

:(*
dtype0u
$svdf_0/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"1   (   �
svdf_0/dense/Tensordot/ReshapeReshapesvdf_0/ExpandDims:output:0-svdf_0/dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:1(�
svdf_0/dense/Tensordot/MatMulMatMul'svdf_0/dense/Tensordot/Reshape:output:0-svdf_0/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:1u
svdf_0/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   1         �
svdf_0/dense/TensordotReshape'svdf_0/dense/Tensordot/MatMul:product:0%svdf_0/dense/Tensordot/shape:output:0*
T0*&
_output_shapes
:1�
svdf_0/stream/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_0/stream/PadPadsvdf_0/dense/Tensordot:output:0#svdf_0/stream/Pad/paddings:output:0*
T0*&
_output_shapes
:1�
7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpVsvdf_0_stream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel*&
_output_shapes
:*
dtype0�
.svdf_0/stream/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
6svdf_0/stream/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
(svdf_0/stream/depthwise_conv2d/depthwiseDepthwiseConv2dNativesvdf_0/stream/Pad:output:0?svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:.*
paddingVALID*
strides
�
5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpHsvdf_0_stream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias*
_output_shapes
:*
dtype0�
&svdf_0/stream/depthwise_conv2d/BiasAddBiasAdd1svdf_0/stream/depthwise_conv2d/depthwise:output:0=svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.u
svdf_0/ReluRelu/svdf_0/stream/depthwise_conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:.�
'svdf_0/dense_1/Tensordot/ReadVariableOpReadVariableOp=svdf_0_dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel*
_output_shapes

:(*
dtype0w
&svdf_0/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".      �
 svdf_0/dense_1/Tensordot/ReshapeReshapesvdf_0/Relu:activations:0/svdf_0/dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.�
svdf_0/dense_1/Tensordot/MatMulMatMul)svdf_0/dense_1/Tensordot/Reshape:output:0/svdf_0/dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:.(w
svdf_0/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .      (   �
svdf_0/dense_1/TensordotReshape)svdf_0/dense_1/Tensordot/MatMul:product:0'svdf_0/dense_1/Tensordot/shape:output:0*
T0*&
_output_shapes
:.(�
%svdf_0/dense_1/BiasAdd/ReadVariableOpReadVariableOp9svdf_0_dense_1_biasadd_readvariableop_svdf_0_dense_1_bias*
_output_shapes
:(*
dtype0�
svdf_0/dense_1/BiasAddBiasAdd!svdf_0/dense_1/Tensordot:output:0-svdf_0/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.(~
svdf_0/SqueezeSqueezesvdf_0/dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:.(*
squeeze_dims
W
svdf_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_1/ExpandDims
ExpandDimssvdf_0/Squeeze:output:0svdf_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:.(�
'svdf_1/dense_2/Tensordot/ReadVariableOpReadVariableOp=svdf_1_dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel*
_output_shapes

:( *
dtype0w
&svdf_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".   (   �
 svdf_1/dense_2/Tensordot/ReshapeReshapesvdf_1/ExpandDims:output:0/svdf_1/dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.(�
svdf_1/dense_2/Tensordot/MatMulMatMul)svdf_1/dense_2/Tensordot/Reshape:output:0/svdf_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:. w
svdf_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .          �
svdf_1/dense_2/TensordotReshape)svdf_1/dense_2/Tensordot/MatMul:product:0'svdf_1/dense_2/Tensordot/shape:output:0*
T0*&
_output_shapes
:. �
svdf_1/stream_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_1/stream_1/PadPad!svdf_1/dense_2/Tensordot:output:0%svdf_1/stream_1/Pad/paddings:output:0*
T0*&
_output_shapes
:. �
;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp\svdf_1_stream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
2svdf_1/stream_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
:svdf_1/stream_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_1/stream_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativesvdf_1/stream_1/Pad:output:0Csvdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:% *
paddingVALID*
strides
�
9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpNsvdf_1_stream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias*
_output_shapes
: *
dtype0�
*svdf_1/stream_1/depthwise_conv2d_1/BiasAddBiasAdd5svdf_1/stream_1/depthwise_conv2d_1/depthwise:output:0Asvdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:% y
svdf_1/ReluRelu3svdf_1/stream_1/depthwise_conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:% �
'svdf_1/dense_3/Tensordot/ReadVariableOpReadVariableOp=svdf_1_dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel*
_output_shapes

: (*
dtype0w
&svdf_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%       �
 svdf_1/dense_3/Tensordot/ReshapeReshapesvdf_1/Relu:activations:0/svdf_1/dense_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:% �
svdf_1/dense_3/Tensordot/MatMulMatMul)svdf_1/dense_3/Tensordot/Reshape:output:0/svdf_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:%(w
svdf_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %      (   �
svdf_1/dense_3/TensordotReshape)svdf_1/dense_3/Tensordot/MatMul:product:0'svdf_1/dense_3/Tensordot/shape:output:0*
T0*&
_output_shapes
:%(�
%svdf_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp9svdf_1_dense_3_biasadd_readvariableop_svdf_1_dense_3_bias*
_output_shapes
:(*
dtype0�
svdf_1/dense_3/BiasAddBiasAdd!svdf_1/dense_3/Tensordot:output:0-svdf_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:%(~
svdf_1/SqueezeSqueezesvdf_1/dense_3/BiasAdd:output:0*
T0*"
_output_shapes
:%(*
squeeze_dims
W
svdf_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_2/ExpandDims
ExpandDimssvdf_1/Squeeze:output:0svdf_2/ExpandDims/dim:output:0*
T0*&
_output_shapes
:%(�
'svdf_2/dense_4/Tensordot/ReadVariableOpReadVariableOp=svdf_2_dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel*
_output_shapes

:( *
dtype0w
&svdf_2/dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%   (   �
 svdf_2/dense_4/Tensordot/ReshapeReshapesvdf_2/ExpandDims:output:0/svdf_2/dense_4/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:%(�
svdf_2/dense_4/Tensordot/MatMulMatMul)svdf_2/dense_4/Tensordot/Reshape:output:0/svdf_2/dense_4/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:% w
svdf_2/dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %          �
svdf_2/dense_4/TensordotReshape)svdf_2/dense_4/Tensordot/MatMul:product:0'svdf_2/dense_4/Tensordot/shape:output:0*
T0*&
_output_shapes
:% �
svdf_2/stream_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_2/stream_2/PadPad!svdf_2/dense_4/Tensordot:output:0%svdf_2/stream_2/Pad/paddings:output:0*
T0*&
_output_shapes
:% �
;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp\svdf_2_stream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
2svdf_2/stream_2/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
:svdf_2/stream_2/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_2/stream_2/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativesvdf_2/stream_2/Pad:output:0Csvdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpNsvdf_2_stream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias*
_output_shapes
: *
dtype0�
*svdf_2/stream_2/depthwise_conv2d_2/BiasAddBiasAdd5svdf_2/stream_2/depthwise_conv2d_2/depthwise:output:0Asvdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: y
svdf_2/ReluRelu3svdf_2/stream_2/depthwise_conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
: �
'svdf_2/dense_5/Tensordot/ReadVariableOpReadVariableOp=svdf_2_dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel*
_output_shapes

: @*
dtype0w
&svdf_2/dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
 svdf_2/dense_5/Tensordot/ReshapeReshapesvdf_2/Relu:activations:0/svdf_2/dense_5/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
svdf_2/dense_5/Tensordot/MatMulMatMul)svdf_2/dense_5/Tensordot/Reshape:output:0/svdf_2/dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@w
svdf_2/dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
svdf_2/dense_5/TensordotReshape)svdf_2/dense_5/Tensordot/MatMul:product:0'svdf_2/dense_5/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
%svdf_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp9svdf_2_dense_5_biasadd_readvariableop_svdf_2_dense_5_bias*
_output_shapes
:@*
dtype0�
svdf_2/dense_5/BiasAddBiasAdd!svdf_2/dense_5/Tensordot:output:0-svdf_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@~
svdf_2/SqueezeSqueezesvdf_2/dense_5/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
W
svdf_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_3/ExpandDims
ExpandDimssvdf_2/Squeeze:output:0svdf_3/ExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
'svdf_3/dense_6/Tensordot/ReadVariableOpReadVariableOp=svdf_3_dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel*
_output_shapes

:@ *
dtype0w
&svdf_3/dense_6/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
 svdf_3/dense_6/Tensordot/ReshapeReshapesvdf_3/ExpandDims:output:0/svdf_3/dense_6/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
svdf_3/dense_6/Tensordot/MatMulMatMul)svdf_3/dense_6/Tensordot/Reshape:output:0/svdf_3/dense_6/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

: w
svdf_3/dense_6/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
svdf_3/dense_6/TensordotReshape)svdf_3/dense_6/Tensordot/MatMul:product:0'svdf_3/dense_6/Tensordot/shape:output:0*
T0*&
_output_shapes
: �
svdf_3/stream_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_3/stream_3/PadPad!svdf_3/dense_6/Tensordot:output:0%svdf_3/stream_3/Pad/paddings:output:0*
T0*&
_output_shapes
: �
;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp\svdf_3_stream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
2svdf_3/stream_3/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
:svdf_3/stream_3/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_3/stream_3/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativesvdf_3/stream_3/Pad:output:0Csvdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpNsvdf_3_stream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias*
_output_shapes
: *
dtype0�
*svdf_3/stream_3/depthwise_conv2d_3/BiasAddBiasAdd5svdf_3/stream_3/depthwise_conv2d_3/depthwise:output:0Asvdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: y
svdf_3/ReluRelu3svdf_3/stream_3/depthwise_conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
: �
'svdf_3/dense_7/Tensordot/ReadVariableOpReadVariableOp=svdf_3_dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel*
_output_shapes

: @*
dtype0w
&svdf_3/dense_7/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
 svdf_3/dense_7/Tensordot/ReshapeReshapesvdf_3/Relu:activations:0/svdf_3/dense_7/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
svdf_3/dense_7/Tensordot/MatMulMatMul)svdf_3/dense_7/Tensordot/Reshape:output:0/svdf_3/dense_7/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@w
svdf_3/dense_7/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
svdf_3/dense_7/TensordotReshape)svdf_3/dense_7/Tensordot/MatMul:product:0'svdf_3/dense_7/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
%svdf_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp9svdf_3_dense_7_biasadd_readvariableop_svdf_3_dense_7_bias*
_output_shapes
:@*
dtype0�
svdf_3/dense_7/BiasAddBiasAdd!svdf_3/dense_7/Tensordot:output:0-svdf_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@~
svdf_3/SqueezeSqueezesvdf_3/dense_7/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
W
svdf_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_4/ExpandDims
ExpandDimssvdf_3/Squeeze:output:0svdf_4/ExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
'svdf_4/dense_8/Tensordot/ReadVariableOpReadVariableOp=svdf_4_dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel*
_output_shapes

:@@*
dtype0w
&svdf_4/dense_8/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
 svdf_4/dense_8/Tensordot/ReshapeReshapesvdf_4/ExpandDims:output:0/svdf_4/dense_8/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
svdf_4/dense_8/Tensordot/MatMulMatMul)svdf_4/dense_8/Tensordot/Reshape:output:0/svdf_4/dense_8/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@w
svdf_4/dense_8/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
svdf_4/dense_8/TensordotReshape)svdf_4/dense_8/Tensordot/MatMul:product:0'svdf_4/dense_8/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
svdf_4/stream_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_4/stream_4/PadPad!svdf_4/dense_8/Tensordot:output:0%svdf_4/stream_4/Pad/paddings:output:0*
T0*&
_output_shapes
:@�
;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOp\svdf_4_stream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel*&
_output_shapes
:
@*
dtype0�
2svdf_4/stream_4/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      @      �
:svdf_4/stream_4/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_4/stream_4/depthwise_conv2d_4/depthwiseDepthwiseConv2dNativesvdf_4/stream_4/Pad:output:0Csvdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@*
paddingVALID*
strides
�
9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpNsvdf_4_stream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias*
_output_shapes
:@*
dtype0�
*svdf_4/stream_4/depthwise_conv2d_4/BiasAddBiasAdd5svdf_4/stream_4/depthwise_conv2d_4/depthwise:output:0Asvdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@y
svdf_4/ReluRelu3svdf_4/stream_4/depthwise_conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
:
@�
'svdf_4/dense_9/Tensordot/ReadVariableOpReadVariableOp=svdf_4_dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel*
_output_shapes

:@@*
dtype0w
&svdf_4/dense_9/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
 svdf_4/dense_9/Tensordot/ReshapeReshapesvdf_4/Relu:activations:0/svdf_4/dense_9/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
svdf_4/dense_9/Tensordot/MatMulMatMul)svdf_4/dense_9/Tensordot/Reshape:output:0/svdf_4/dense_9/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:
@w
svdf_4/dense_9/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      @   �
svdf_4/dense_9/TensordotReshape)svdf_4/dense_9/Tensordot/MatMul:product:0'svdf_4/dense_9/Tensordot/shape:output:0*
T0*&
_output_shapes
:
@�
%svdf_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp9svdf_4_dense_9_biasadd_readvariableop_svdf_4_dense_9_bias*
_output_shapes
:@*
dtype0�
svdf_4/dense_9/BiasAddBiasAdd!svdf_4/dense_9/Tensordot:output:0-svdf_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@~
svdf_4/SqueezeSqueezesvdf_4/dense_9/BiasAdd:output:0*
T0*"
_output_shapes
:
@*
squeeze_dims
W
svdf_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_5/ExpandDims
ExpandDimssvdf_4/Squeeze:output:0svdf_5/ExpandDims/dim:output:0*
T0*&
_output_shapes
:
@�
(svdf_5/dense_10/Tensordot/ReadVariableOpReadVariableOp?svdf_5_dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel*
_output_shapes
:	@�*
dtype0x
'svdf_5/dense_10/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
!svdf_5/dense_10/Tensordot/ReshapeReshapesvdf_5/ExpandDims:output:00svdf_5/dense_10/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
 svdf_5/dense_10/Tensordot/MatMulMatMul*svdf_5/dense_10/Tensordot/Reshape:output:00svdf_5/dense_10/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	
�x
svdf_5/dense_10/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      �   �
svdf_5/dense_10/TensordotReshape*svdf_5/dense_10/Tensordot/MatMul:product:0(svdf_5/dense_10/Tensordot/shape:output:0*
T0*'
_output_shapes
:
��
svdf_5/stream_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_5/stream_5/PadPad"svdf_5/dense_10/Tensordot:output:0%svdf_5/stream_5/Pad/paddings:output:0*
T0*'
_output_shapes
:
��
;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOp\svdf_5_stream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel*'
_output_shapes
:
�*
dtype0�
2svdf_5/stream_5/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      �      �
:svdf_5/stream_5/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_5/stream_5/depthwise_conv2d_5/depthwiseDepthwiseConv2dNativesvdf_5/stream_5/Pad:output:0Csvdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpReadVariableOpNsvdf_5_stream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias*
_output_shapes	
:�*
dtype0�
*svdf_5/stream_5/depthwise_conv2d_5/BiasAddBiasAdd5svdf_5/stream_5/depthwise_conv2d_5/depthwise:output:0Asvdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�z
svdf_5/ReluRelu3svdf_5/stream_5/depthwise_conv2d_5/BiasAdd:output:0*
T0*'
_output_shapes
:�y
svdf_5/SqueezeSqueezesvdf_5/Relu:activations:0*
T0*#
_output_shapes
:�*
squeeze_dims
g
stream_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
stream_6/flatten/ReshapeReshapesvdf_5/Squeeze:output:0stream_6/flatten/Const:output:0*
T0*
_output_shapes
:	�i
dropout/IdentityIdentity!stream_6/flatten/Reshape:output:0*
T0*
_output_shapes
:	��
dense_11/MatMul/ReadVariableOpReadVariableOp.dense_11_matmul_readvariableop_dense_11_kernel*
_output_shapes
:	�*
dtype0�
dense_11/MatMulMatMuldropout/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
dense_11/BiasAdd/ReadVariableOpReadVariableOp-dense_11_biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp&^svdf_0/dense/Tensordot/ReadVariableOp&^svdf_0/dense_1/BiasAdd/ReadVariableOp(^svdf_0/dense_1/Tensordot/ReadVariableOp6^svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp8^svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp(^svdf_1/dense_2/Tensordot/ReadVariableOp&^svdf_1/dense_3/BiasAdd/ReadVariableOp(^svdf_1/dense_3/Tensordot/ReadVariableOp:^svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp<^svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp(^svdf_2/dense_4/Tensordot/ReadVariableOp&^svdf_2/dense_5/BiasAdd/ReadVariableOp(^svdf_2/dense_5/Tensordot/ReadVariableOp:^svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp<^svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp(^svdf_3/dense_6/Tensordot/ReadVariableOp&^svdf_3/dense_7/BiasAdd/ReadVariableOp(^svdf_3/dense_7/Tensordot/ReadVariableOp:^svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp<^svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp(^svdf_4/dense_8/Tensordot/ReadVariableOp&^svdf_4/dense_9/BiasAdd/ReadVariableOp(^svdf_4/dense_9/Tensordot/ReadVariableOp:^svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp<^svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp)^svdf_5/dense_10/Tensordot/ReadVariableOp:^svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp<^svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2N
%svdf_0/dense/Tensordot/ReadVariableOp%svdf_0/dense/Tensordot/ReadVariableOp2N
%svdf_0/dense_1/BiasAdd/ReadVariableOp%svdf_0/dense_1/BiasAdd/ReadVariableOp2R
'svdf_0/dense_1/Tensordot/ReadVariableOp'svdf_0/dense_1/Tensordot/ReadVariableOp2n
5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp2r
7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp2R
'svdf_1/dense_2/Tensordot/ReadVariableOp'svdf_1/dense_2/Tensordot/ReadVariableOp2N
%svdf_1/dense_3/BiasAdd/ReadVariableOp%svdf_1/dense_3/BiasAdd/ReadVariableOp2R
'svdf_1/dense_3/Tensordot/ReadVariableOp'svdf_1/dense_3/Tensordot/ReadVariableOp2v
9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2z
;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp2R
'svdf_2/dense_4/Tensordot/ReadVariableOp'svdf_2/dense_4/Tensordot/ReadVariableOp2N
%svdf_2/dense_5/BiasAdd/ReadVariableOp%svdf_2/dense_5/BiasAdd/ReadVariableOp2R
'svdf_2/dense_5/Tensordot/ReadVariableOp'svdf_2/dense_5/Tensordot/ReadVariableOp2v
9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2z
;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp2R
'svdf_3/dense_6/Tensordot/ReadVariableOp'svdf_3/dense_6/Tensordot/ReadVariableOp2N
%svdf_3/dense_7/BiasAdd/ReadVariableOp%svdf_3/dense_7/BiasAdd/ReadVariableOp2R
'svdf_3/dense_7/Tensordot/ReadVariableOp'svdf_3/dense_7/Tensordot/ReadVariableOp2v
9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2z
;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp2R
'svdf_4/dense_8/Tensordot/ReadVariableOp'svdf_4/dense_8/Tensordot/ReadVariableOp2N
%svdf_4/dense_9/BiasAdd/ReadVariableOp%svdf_4/dense_9/BiasAdd/ReadVariableOp2R
'svdf_4/dense_9/Tensordot/ReadVariableOp'svdf_4/dense_9/Tensordot/ReadVariableOp2v
9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2z
;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp2T
(svdf_5/dense_10/Tensordot/ReadVariableOp(svdf_5/dense_10/Tensordot/ReadVariableOp2v
9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2z
;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�
�

"__inference_signature_wrapper_3795
input_1%
svdf_0_dense_kernel:(8
svdf_0_stream_depthwise_kernel: 
svdf_0_stream_bias:'
svdf_0_dense_1_kernel:(!
svdf_0_dense_1_bias:('
svdf_1_dense_2_kernel:( :
 svdf_1_stream_1_depthwise_kernel:
 "
svdf_1_stream_1_bias: '
svdf_1_dense_3_kernel: (!
svdf_1_dense_3_bias:('
svdf_2_dense_4_kernel:( :
 svdf_2_stream_2_depthwise_kernel:
 "
svdf_2_stream_2_bias: '
svdf_2_dense_5_kernel: @!
svdf_2_dense_5_bias:@'
svdf_3_dense_6_kernel:@ :
 svdf_3_stream_3_depthwise_kernel:
 "
svdf_3_stream_3_bias: '
svdf_3_dense_7_kernel: @!
svdf_3_dense_7_bias:@'
svdf_4_dense_8_kernel:@@:
 svdf_4_stream_4_depthwise_kernel:
@"
svdf_4_stream_4_bias:@'
svdf_4_dense_9_kernel:@@!
svdf_4_dense_9_bias:@)
svdf_5_dense_10_kernel:	@�;
 svdf_5_stream_5_depthwise_kernel:
�#
svdf_5_stream_5_bias:	�"
dense_11_kernel:	�
dense_11_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1svdf_0_dense_kernelsvdf_0_stream_depthwise_kernelsvdf_0_stream_biassvdf_0_dense_1_kernelsvdf_0_dense_1_biassvdf_1_dense_2_kernel svdf_1_stream_1_depthwise_kernelsvdf_1_stream_1_biassvdf_1_dense_3_kernelsvdf_1_dense_3_biassvdf_2_dense_4_kernel svdf_2_stream_2_depthwise_kernelsvdf_2_stream_2_biassvdf_2_dense_5_kernelsvdf_2_dense_5_biassvdf_3_dense_6_kernel svdf_3_stream_3_depthwise_kernelsvdf_3_stream_3_biassvdf_3_dense_7_kernelsvdf_3_dense_7_biassvdf_4_dense_8_kernel svdf_4_stream_4_depthwise_kernelsvdf_4_stream_4_biassvdf_4_dense_9_kernelsvdf_4_dense_9_biassvdf_5_dense_10_kernel svdf_5_stream_5_depthwise_kernelsvdf_5_stream_5_biasdense_11_kerneldense_11_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_2280f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	�}
!
_user_specified_name	input_1
�	
�
%__inference_svdf_4_layer_call_fn_4677

inputs'
svdf_4_dense_8_kernel:@@:
 svdf_4_stream_4_depthwise_kernel:
@"
svdf_4_stream_4_bias:@'
svdf_4_dense_9_kernel:@@!
svdf_4_dense_9_bias:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_4_dense_8_kernel svdf_4_stream_4_depthwise_kernelsvdf_4_stream_4_biassvdf_4_dense_9_kernelsvdf_4_dense_9_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:
@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_4_layer_call_and_return_conditional_losses_2801j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:
@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�	
�
%__inference_svdf_4_layer_call_fn_4667

inputs'
svdf_4_dense_8_kernel:@@:
 svdf_4_stream_4_depthwise_kernel:
@"
svdf_4_stream_4_bias:@'
svdf_4_dense_9_kernel:@@!
svdf_4_dense_9_bias:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_4_dense_8_kernel svdf_4_stream_4_depthwise_kernelsvdf_4_stream_4_biassvdf_4_dense_9_kernelsvdf_4_dense_9_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:
@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_4_layer_call_and_return_conditional_losses_2505j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:
@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_4844

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	�S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0*
_input_shapes
:	�:G C

_output_shapes
:	�
 
_user_specified_nameinputs
��
�
 __inference__traced_restore_5074
file_prefix3
 assignvariableop_dense_11_kernel:	�.
 assignvariableop_1_dense_11_bias:8
&assignvariableop_2_svdf_0_dense_kernel:(K
1assignvariableop_3_svdf_0_stream_depthwise_kernel:3
%assignvariableop_4_svdf_0_stream_bias::
(assignvariableop_5_svdf_0_dense_1_kernel:(4
&assignvariableop_6_svdf_0_dense_1_bias:(:
(assignvariableop_7_svdf_1_dense_2_kernel:( M
3assignvariableop_8_svdf_1_stream_1_depthwise_kernel:
 5
'assignvariableop_9_svdf_1_stream_1_bias: ;
)assignvariableop_10_svdf_1_dense_3_kernel: (5
'assignvariableop_11_svdf_1_dense_3_bias:(;
)assignvariableop_12_svdf_2_dense_4_kernel:( N
4assignvariableop_13_svdf_2_stream_2_depthwise_kernel:
 6
(assignvariableop_14_svdf_2_stream_2_bias: ;
)assignvariableop_15_svdf_2_dense_5_kernel: @5
'assignvariableop_16_svdf_2_dense_5_bias:@;
)assignvariableop_17_svdf_3_dense_6_kernel:@ N
4assignvariableop_18_svdf_3_stream_3_depthwise_kernel:
 6
(assignvariableop_19_svdf_3_stream_3_bias: ;
)assignvariableop_20_svdf_3_dense_7_kernel: @5
'assignvariableop_21_svdf_3_dense_7_bias:@;
)assignvariableop_22_svdf_4_dense_8_kernel:@@N
4assignvariableop_23_svdf_4_stream_4_depthwise_kernel:
@6
(assignvariableop_24_svdf_4_stream_4_bias:@;
)assignvariableop_25_svdf_4_dense_9_kernel:@@5
'assignvariableop_26_svdf_4_dense_9_bias:@=
*assignvariableop_27_svdf_5_dense_10_kernel:	@�O
4assignvariableop_28_svdf_5_stream_5_depthwise_kernel:
�7
(assignvariableop_29_svdf_5_stream_5_bias:	�
identity_31��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp&assignvariableop_2_svdf_0_dense_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp1assignvariableop_3_svdf_0_stream_depthwise_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_svdf_0_stream_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp(assignvariableop_5_svdf_0_dense_1_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_svdf_0_dense_1_biasIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_svdf_1_dense_2_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp3assignvariableop_8_svdf_1_stream_1_depthwise_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp'assignvariableop_9_svdf_1_stream_1_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp)assignvariableop_10_svdf_1_dense_3_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_svdf_1_dense_3_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_svdf_2_dense_4_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp4assignvariableop_13_svdf_2_stream_2_depthwise_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp(assignvariableop_14_svdf_2_stream_2_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp)assignvariableop_15_svdf_2_dense_5_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_svdf_2_dense_5_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp)assignvariableop_17_svdf_3_dense_6_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_svdf_3_stream_3_depthwise_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp(assignvariableop_19_svdf_3_stream_3_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_svdf_3_dense_7_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_svdf_3_dense_7_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_svdf_4_dense_8_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp4assignvariableop_23_svdf_4_stream_4_depthwise_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_svdf_4_stream_4_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_svdf_4_dense_9_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_svdf_4_dense_9_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_svdf_5_dense_10_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp4assignvariableop_28_svdf_5_stream_5_depthwise_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp(assignvariableop_29_svdf_5_stream_5_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_31IdentityIdentity_30:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_31Identity_31:output:0*Q
_input_shapes@
>: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
B__inference_dense_11_layer_call_and_return_conditional_losses_4861

inputs8
%matmul_readvariableop_dense_11_kernel:	�2
$biasadd_readvariableop_dense_11_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_11_kernel*
_output_shapes
:	�*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*"
_input_shapes
:	�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�	
�
%__inference_svdf_0_layer_call_fn_4317

inputs%
svdf_0_dense_kernel:(8
svdf_0_stream_depthwise_kernel: 
svdf_0_stream_bias:'
svdf_0_dense_1_kernel:(!
svdf_0_dense_1_bias:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_0_dense_kernelsvdf_0_stream_depthwise_kernelsvdf_0_stream_biassvdf_0_dense_1_kernelsvdf_0_dense_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:.(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_0_layer_call_and_return_conditional_losses_3305j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:.(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:1(: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
�*
�
@__inference_svdf_4_layer_call_and_return_conditional_losses_4747

inputsH
6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel:@@o
Ustream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel:
@U
Gstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias:@H
6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel:@@@
2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias:@
identity�� dense_8/Tensordot/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp� dense_9/Tensordot/ReadVariableOp�2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp�4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_8/Tensordot/ReadVariableOpReadVariableOp6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel*
_output_shapes

:@@*
dtype0p
dense_8/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_8/Tensordot/ReshapeReshapeExpandDims:output:0(dense_8/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_8/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0 dense_8/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
stream_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_4/PadPaddense_8/Tensordot:output:0stream_4/Pad/paddings:output:0*
T0*&
_output_shapes
:@�
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOpUstream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel*&
_output_shapes
:
@*
dtype0�
+stream_4/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      @      �
3stream_4/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_4/depthwise_conv2d_4/depthwiseDepthwiseConv2dNativestream_4/Pad:output:0<stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@*
paddingVALID*
strides
�
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpGstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias*
_output_shapes
:@*
dtype0�
#stream_4/depthwise_conv2d_4/BiasAddBiasAdd.stream_4/depthwise_conv2d_4/depthwise:output:0:stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@k
ReluRelu,stream_4/depthwise_conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
:
@�
 dense_9/Tensordot/ReadVariableOpReadVariableOp6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel*
_output_shapes

:@@*
dtype0p
dense_9/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_9/Tensordot/ReshapeReshapeRelu:activations:0(dense_9/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:
@p
dense_9/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      @   �
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0 dense_9/Tensordot/shape:output:0*
T0*&
_output_shapes
:
@�
dense_9/BiasAdd/ReadVariableOpReadVariableOp2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias*
_output_shapes
:@*
dtype0�
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@p
SqueezeSqueezedense_9/BiasAdd:output:0*
T0*"
_output_shapes
:
@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:
@�
NoOpNoOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp3^stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp5^stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2h
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2l
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�
^
B__inference_stream_6_layer_call_and_return_conditional_losses_2548

inputs
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   d
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	�X
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:�:K G
#
_output_shapes
:�
 
_user_specified_nameinputs
�
B
&__inference_dropout_layer_call_fn_4829

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2555X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*
_input_shapes
:	�:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�
�
@__inference_svdf_5_layer_call_and_return_conditional_losses_4813

inputsK
8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel:	@�p
Ustream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel:
�V
Gstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias:	�
identity��!dense_10/Tensordot/ReadVariableOp�2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp�4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:
@�
!dense_10/Tensordot/ReadVariableOpReadVariableOp8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel*
_output_shapes
:	@�*
dtype0q
 dense_10/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_10/Tensordot/ReshapeReshapeExpandDims:output:0)dense_10/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	
�q
dense_10/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      �   �
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0!dense_10/Tensordot/shape:output:0*
T0*'
_output_shapes
:
��
stream_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_5/PadPaddense_10/Tensordot:output:0stream_5/Pad/paddings:output:0*
T0*'
_output_shapes
:
��
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOpUstream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel*'
_output_shapes
:
�*
dtype0�
+stream_5/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      �      �
3stream_5/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_5/depthwise_conv2d_5/depthwiseDepthwiseConv2dNativestream_5/Pad:output:0<stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpReadVariableOpGstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias*
_output_shapes	
:�*
dtype0�
#stream_5/depthwise_conv2d_5/BiasAddBiasAdd.stream_5/depthwise_conv2d_5/depthwise:output:0:stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�l
ReluRelu,stream_5/depthwise_conv2d_5/BiasAdd:output:0*
T0*'
_output_shapes
:�k
SqueezeSqueezeRelu:activations:0*
T0*#
_output_shapes
:�*
squeeze_dims
[
IdentityIdentitySqueeze:output:0^NoOp*
T0*#
_output_shapes
:��
NoOpNoOp"^dense_10/Tensordot/ReadVariableOp3^stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp5^stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:
@: : : 2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2h
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2l
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:J F
"
_output_shapes
:
@
 
_user_specified_nameinputs
�*
�
@__inference_svdf_4_layer_call_and_return_conditional_losses_2801

inputsH
6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel:@@o
Ustream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel:
@U
Gstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias:@H
6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel:@@@
2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias:@
identity�� dense_8/Tensordot/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp� dense_9/Tensordot/ReadVariableOp�2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp�4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_8/Tensordot/ReadVariableOpReadVariableOp6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel*
_output_shapes

:@@*
dtype0p
dense_8/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_8/Tensordot/ReshapeReshapeExpandDims:output:0(dense_8/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_8/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0 dense_8/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
stream_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_4/PadPaddense_8/Tensordot:output:0stream_4/Pad/paddings:output:0*
T0*&
_output_shapes
:@�
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOpUstream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel*&
_output_shapes
:
@*
dtype0�
+stream_4/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      @      �
3stream_4/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_4/depthwise_conv2d_4/depthwiseDepthwiseConv2dNativestream_4/Pad:output:0<stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@*
paddingVALID*
strides
�
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpGstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias*
_output_shapes
:@*
dtype0�
#stream_4/depthwise_conv2d_4/BiasAddBiasAdd.stream_4/depthwise_conv2d_4/depthwise:output:0:stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@k
ReluRelu,stream_4/depthwise_conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
:
@�
 dense_9/Tensordot/ReadVariableOpReadVariableOp6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel*
_output_shapes

:@@*
dtype0p
dense_9/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_9/Tensordot/ReshapeReshapeRelu:activations:0(dense_9/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:
@p
dense_9/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      @   �
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0 dense_9/Tensordot/shape:output:0*
T0*&
_output_shapes
:
@�
dense_9/BiasAdd/ReadVariableOpReadVariableOp2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias*
_output_shapes
:@*
dtype0�
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@p
SqueezeSqueezedense_9/BiasAdd:output:0*
T0*"
_output_shapes
:
@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:
@�
NoOpNoOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp3^stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp5^stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:@: : : : : 2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2h
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2l
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�*
�
@__inference_svdf_1_layer_call_and_return_conditional_losses_3179

inputsH
6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel:( o
Ustream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel:
 U
Gstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias: H
6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel: (@
2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias:(
identity�� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp�4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:.(�
 dense_2/Tensordot/ReadVariableOpReadVariableOp6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel*
_output_shapes

:( *
dtype0p
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".   (   �
dense_2/Tensordot/ReshapeReshapeExpandDims:output:0(dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.(�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:. p
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .          �
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*&
_output_shapes
:. �
stream_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_1/PadPaddense_2/Tensordot:output:0stream_1/Pad/paddings:output:0*
T0*&
_output_shapes
:. �
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpUstream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativestream_1/Pad:output:0<stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:% *
paddingVALID*
strides
�
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias*
_output_shapes
: *
dtype0�
#stream_1/depthwise_conv2d_1/BiasAddBiasAdd.stream_1/depthwise_conv2d_1/depthwise:output:0:stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:% k
ReluRelu,stream_1/depthwise_conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:% �
 dense_3/Tensordot/ReadVariableOpReadVariableOp6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel*
_output_shapes

: (*
dtype0p
dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%       �
dense_3/Tensordot/ReshapeReshapeRelu:activations:0(dense_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:% �
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:%(p
dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %      (   �
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0 dense_3/Tensordot/shape:output:0*
T0*&
_output_shapes
:%(�
dense_3/BiasAdd/ReadVariableOpReadVariableOp2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias*
_output_shapes
:(*
dtype0�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:%(p
SqueezeSqueezedense_3/BiasAdd:output:0*
T0*"
_output_shapes
:%(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:%(�
NoOpNoOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp3^stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp5^stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:.(: : : : : 2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2h
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2l
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:J F
"
_output_shapes
:.(
 
_user_specified_nameinputs
�
^
B__inference_stream_6_layer_call_and_return_conditional_losses_4824

inputs
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   d
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	�X
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*"
_input_shapes
:�:K G
#
_output_shapes
:�
 
_user_specified_nameinputs
�
J
.__inference_speech_features_layer_call_fn_4271

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_speech_features_layer_call_and_return_conditional_losses_3404[
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:1("
identityIdentity:output:0*
_input_shapes
:	�}:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�*
�
@__inference_svdf_1_layer_call_and_return_conditional_losses_2379

inputsH
6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel:( o
Ustream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel:
 U
Gstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias: H
6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel: (@
2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias:(
identity�� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp�4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:.(�
 dense_2/Tensordot/ReadVariableOpReadVariableOp6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel*
_output_shapes

:( *
dtype0p
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".   (   �
dense_2/Tensordot/ReshapeReshapeExpandDims:output:0(dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.(�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:. p
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .          �
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*&
_output_shapes
:. �
stream_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_1/PadPaddense_2/Tensordot:output:0stream_1/Pad/paddings:output:0*
T0*&
_output_shapes
:. �
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpUstream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativestream_1/Pad:output:0<stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:% *
paddingVALID*
strides
�
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias*
_output_shapes
: *
dtype0�
#stream_1/depthwise_conv2d_1/BiasAddBiasAdd.stream_1/depthwise_conv2d_1/depthwise:output:0:stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:% k
ReluRelu,stream_1/depthwise_conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:% �
 dense_3/Tensordot/ReadVariableOpReadVariableOp6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel*
_output_shapes

: (*
dtype0p
dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%       �
dense_3/Tensordot/ReshapeReshapeRelu:activations:0(dense_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:% �
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:%(p
dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %      (   �
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0 dense_3/Tensordot/shape:output:0*
T0*&
_output_shapes
:%(�
dense_3/BiasAdd/ReadVariableOpReadVariableOp2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias*
_output_shapes
:(*
dtype0�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:%(p
SqueezeSqueezedense_3/BiasAdd:output:0*
T0*"
_output_shapes
:%(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:%(�
NoOpNoOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp3^stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp5^stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:.(: : : : : 2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2h
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2l
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:J F
"
_output_shapes
:.(
 
_user_specified_nameinputs
�)
�
@__inference_svdf_0_layer_call_and_return_conditional_losses_4387

inputsD
2dense_tensordot_readvariableop_svdf_0_dense_kernel:(i
Ostream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel:O
Astream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias:H
6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel:(@
2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias:(
identity��dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�.stream/depthwise_conv2d/BiasAdd/ReadVariableOp�0stream/depthwise_conv2d/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:1(�
dense/Tensordot/ReadVariableOpReadVariableOp2dense_tensordot_readvariableop_svdf_0_dense_kernel*
_output_shapes

:(*
dtype0n
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"1   (   �
dense/Tensordot/ReshapeReshapeExpandDims:output:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:1(�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:1n
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   1         �
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*&
_output_shapes
:1�
stream/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 z

stream/PadPaddense/Tensordot:output:0stream/Pad/paddings:output:0*
T0*&
_output_shapes
:1�
0stream/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpOstream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel*&
_output_shapes
:*
dtype0�
'stream/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
/stream/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
!stream/depthwise_conv2d/depthwiseDepthwiseConv2dNativestream/Pad:output:08stream/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:.*
paddingVALID*
strides
�
.stream/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpAstream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias*
_output_shapes
:*
dtype0�
stream/depthwise_conv2d/BiasAddBiasAdd*stream/depthwise_conv2d/depthwise:output:06stream/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.g
ReluRelu(stream/depthwise_conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:.�
 dense_1/Tensordot/ReadVariableOpReadVariableOp6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel*
_output_shapes

:(*
dtype0p
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".      �
dense_1/Tensordot/ReshapeReshapeRelu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:.(p
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .      (   �
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*&
_output_shapes
:.(�
dense_1/BiasAdd/ReadVariableOpReadVariableOp2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias*
_output_shapes
:(*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.(p
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*"
_output_shapes
:.(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:.(�
NoOpNoOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^stream/depthwise_conv2d/BiasAdd/ReadVariableOp1^stream/depthwise_conv2d/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:1(: : : : : 2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2`
.stream/depthwise_conv2d/BiasAdd/ReadVariableOp.stream/depthwise_conv2d/BiasAdd/ReadVariableOp2d
0stream/depthwise_conv2d/depthwise/ReadVariableOp0stream/depthwise_conv2d/depthwise/ReadVariableOp:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
�*
�
@__inference_svdf_3_layer_call_and_return_conditional_losses_4622

inputsH
6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel:@ o
Ustream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel:
 U
Gstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias: H
6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel: @@
2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias:@
identity�� dense_6/Tensordot/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp� dense_7/Tensordot/ReadVariableOp�2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp�4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_6/Tensordot/ReadVariableOpReadVariableOp6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel*
_output_shapes

:@ *
dtype0p
dense_6/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_6/Tensordot/ReshapeReshapeExpandDims:output:0(dense_6/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

: p
dense_6/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0 dense_6/Tensordot/shape:output:0*
T0*&
_output_shapes
: �
stream_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_3/PadPaddense_6/Tensordot:output:0stream_3/Pad/paddings:output:0*
T0*&
_output_shapes
: �
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpUstream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_3/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_3/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_3/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativestream_3/Pad:output:0<stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpGstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias*
_output_shapes
: *
dtype0�
#stream_3/depthwise_conv2d_3/BiasAddBiasAdd.stream_3/depthwise_conv2d_3/depthwise:output:0:stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_3/depthwise_conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_7/Tensordot/ReadVariableOpReadVariableOp6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel*
_output_shapes

: @*
dtype0p
dense_7/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_7/Tensordot/ReshapeReshapeRelu:activations:0(dense_7/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_7/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0 dense_7/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_7/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp3^stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp5^stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2h
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2l
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�
� 
?__inference_model_layer_call_and_return_conditional_losses_4063

inputsK
9svdf_0_dense_tensordot_readvariableop_svdf_0_dense_kernel:(p
Vsvdf_0_stream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel:V
Hsvdf_0_stream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias:O
=svdf_0_dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel:(G
9svdf_0_dense_1_biasadd_readvariableop_svdf_0_dense_1_bias:(O
=svdf_1_dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel:( v
\svdf_1_stream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel:
 \
Nsvdf_1_stream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias: O
=svdf_1_dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel: (G
9svdf_1_dense_3_biasadd_readvariableop_svdf_1_dense_3_bias:(O
=svdf_2_dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel:( v
\svdf_2_stream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel:
 \
Nsvdf_2_stream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias: O
=svdf_2_dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel: @G
9svdf_2_dense_5_biasadd_readvariableop_svdf_2_dense_5_bias:@O
=svdf_3_dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel:@ v
\svdf_3_stream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel:
 \
Nsvdf_3_stream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias: O
=svdf_3_dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel: @G
9svdf_3_dense_7_biasadd_readvariableop_svdf_3_dense_7_bias:@O
=svdf_4_dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel:@@v
\svdf_4_stream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel:
@\
Nsvdf_4_stream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias:@O
=svdf_4_dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel:@@G
9svdf_4_dense_9_biasadd_readvariableop_svdf_4_dense_9_bias:@R
?svdf_5_dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel:	@�w
\svdf_5_stream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel:
�]
Nsvdf_5_stream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias:	�A
.dense_11_matmul_readvariableop_dense_11_kernel:	�;
-dense_11_biasadd_readvariableop_dense_11_bias:
identity��dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�%svdf_0/dense/Tensordot/ReadVariableOp�%svdf_0/dense_1/BiasAdd/ReadVariableOp�'svdf_0/dense_1/Tensordot/ReadVariableOp�5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp�7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp�'svdf_1/dense_2/Tensordot/ReadVariableOp�%svdf_1/dense_3/BiasAdd/ReadVariableOp�'svdf_1/dense_3/Tensordot/ReadVariableOp�9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp�;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp�'svdf_2/dense_4/Tensordot/ReadVariableOp�%svdf_2/dense_5/BiasAdd/ReadVariableOp�'svdf_2/dense_5/Tensordot/ReadVariableOp�9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp�;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp�'svdf_3/dense_6/Tensordot/ReadVariableOp�%svdf_3/dense_7/BiasAdd/ReadVariableOp�'svdf_3/dense_7/Tensordot/ReadVariableOp�9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp�;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp�'svdf_4/dense_8/Tensordot/ReadVariableOp�%svdf_4/dense_9/BiasAdd/ReadVariableOp�'svdf_4/dense_9/Tensordot/ReadVariableOp�9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp�;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp�(svdf_5/dense_10/Tensordot/ReadVariableOp�9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp�;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpo
speech_features/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
speech_features/transpose	Transposeinputs'speech_features/transpose/perm:output:0*
T0*
_output_shapes
:	�}�
 speech_features/AudioSpectrogramAudioSpectrogramspeech_features/transpose:y:0*#
_output_shapes
:1�*
magnitude_squared(*
stride�*
window_size�c
 speech_features/Mfcc/sample_rateConst*
_output_shapes
: *
dtype0*
value
B :�}�
speech_features/MfccMfcc.speech_features/AudioSpectrogram:spectrogram:0)speech_features/Mfcc/sample_rate:output:0*"
_output_shapes
:1(*
dct_coefficient_count(*
filterbank_channel_countP*
upper_frequency_limit% ��E�
 speech_features/normalizer/sub/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�                                                                                                                                                                �
speech_features/normalizer/subSubspeech_features/Mfcc:output:0)speech_features/normalizer/sub/y:output:0*
T0*"
_output_shapes
:1(�
$speech_features/normalizer/truediv/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?�
"speech_features/normalizer/truedivRealDiv"speech_features/normalizer/sub:z:0-speech_features/normalizer/truediv/y:output:0*
T0*"
_output_shapes
:1(W
svdf_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_0/ExpandDims
ExpandDims&speech_features/normalizer/truediv:z:0svdf_0/ExpandDims/dim:output:0*
T0*&
_output_shapes
:1(�
%svdf_0/dense/Tensordot/ReadVariableOpReadVariableOp9svdf_0_dense_tensordot_readvariableop_svdf_0_dense_kernel*
_output_shapes

:(*
dtype0u
$svdf_0/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"1   (   �
svdf_0/dense/Tensordot/ReshapeReshapesvdf_0/ExpandDims:output:0-svdf_0/dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:1(�
svdf_0/dense/Tensordot/MatMulMatMul'svdf_0/dense/Tensordot/Reshape:output:0-svdf_0/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:1u
svdf_0/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   1         �
svdf_0/dense/TensordotReshape'svdf_0/dense/Tensordot/MatMul:product:0%svdf_0/dense/Tensordot/shape:output:0*
T0*&
_output_shapes
:1�
svdf_0/stream/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_0/stream/PadPadsvdf_0/dense/Tensordot:output:0#svdf_0/stream/Pad/paddings:output:0*
T0*&
_output_shapes
:1�
7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpVsvdf_0_stream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel*&
_output_shapes
:*
dtype0�
.svdf_0/stream/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
6svdf_0/stream/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
(svdf_0/stream/depthwise_conv2d/depthwiseDepthwiseConv2dNativesvdf_0/stream/Pad:output:0?svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:.*
paddingVALID*
strides
�
5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpHsvdf_0_stream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias*
_output_shapes
:*
dtype0�
&svdf_0/stream/depthwise_conv2d/BiasAddBiasAdd1svdf_0/stream/depthwise_conv2d/depthwise:output:0=svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.u
svdf_0/ReluRelu/svdf_0/stream/depthwise_conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:.�
'svdf_0/dense_1/Tensordot/ReadVariableOpReadVariableOp=svdf_0_dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel*
_output_shapes

:(*
dtype0w
&svdf_0/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".      �
 svdf_0/dense_1/Tensordot/ReshapeReshapesvdf_0/Relu:activations:0/svdf_0/dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.�
svdf_0/dense_1/Tensordot/MatMulMatMul)svdf_0/dense_1/Tensordot/Reshape:output:0/svdf_0/dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:.(w
svdf_0/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .      (   �
svdf_0/dense_1/TensordotReshape)svdf_0/dense_1/Tensordot/MatMul:product:0'svdf_0/dense_1/Tensordot/shape:output:0*
T0*&
_output_shapes
:.(�
%svdf_0/dense_1/BiasAdd/ReadVariableOpReadVariableOp9svdf_0_dense_1_biasadd_readvariableop_svdf_0_dense_1_bias*
_output_shapes
:(*
dtype0�
svdf_0/dense_1/BiasAddBiasAdd!svdf_0/dense_1/Tensordot:output:0-svdf_0/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.(~
svdf_0/SqueezeSqueezesvdf_0/dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:.(*
squeeze_dims
W
svdf_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_1/ExpandDims
ExpandDimssvdf_0/Squeeze:output:0svdf_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:.(�
'svdf_1/dense_2/Tensordot/ReadVariableOpReadVariableOp=svdf_1_dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel*
_output_shapes

:( *
dtype0w
&svdf_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".   (   �
 svdf_1/dense_2/Tensordot/ReshapeReshapesvdf_1/ExpandDims:output:0/svdf_1/dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.(�
svdf_1/dense_2/Tensordot/MatMulMatMul)svdf_1/dense_2/Tensordot/Reshape:output:0/svdf_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:. w
svdf_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .          �
svdf_1/dense_2/TensordotReshape)svdf_1/dense_2/Tensordot/MatMul:product:0'svdf_1/dense_2/Tensordot/shape:output:0*
T0*&
_output_shapes
:. �
svdf_1/stream_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_1/stream_1/PadPad!svdf_1/dense_2/Tensordot:output:0%svdf_1/stream_1/Pad/paddings:output:0*
T0*&
_output_shapes
:. �
;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp\svdf_1_stream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
2svdf_1/stream_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
:svdf_1/stream_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_1/stream_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativesvdf_1/stream_1/Pad:output:0Csvdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:% *
paddingVALID*
strides
�
9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpNsvdf_1_stream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias*
_output_shapes
: *
dtype0�
*svdf_1/stream_1/depthwise_conv2d_1/BiasAddBiasAdd5svdf_1/stream_1/depthwise_conv2d_1/depthwise:output:0Asvdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:% y
svdf_1/ReluRelu3svdf_1/stream_1/depthwise_conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:% �
'svdf_1/dense_3/Tensordot/ReadVariableOpReadVariableOp=svdf_1_dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel*
_output_shapes

: (*
dtype0w
&svdf_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%       �
 svdf_1/dense_3/Tensordot/ReshapeReshapesvdf_1/Relu:activations:0/svdf_1/dense_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:% �
svdf_1/dense_3/Tensordot/MatMulMatMul)svdf_1/dense_3/Tensordot/Reshape:output:0/svdf_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:%(w
svdf_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %      (   �
svdf_1/dense_3/TensordotReshape)svdf_1/dense_3/Tensordot/MatMul:product:0'svdf_1/dense_3/Tensordot/shape:output:0*
T0*&
_output_shapes
:%(�
%svdf_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp9svdf_1_dense_3_biasadd_readvariableop_svdf_1_dense_3_bias*
_output_shapes
:(*
dtype0�
svdf_1/dense_3/BiasAddBiasAdd!svdf_1/dense_3/Tensordot:output:0-svdf_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:%(~
svdf_1/SqueezeSqueezesvdf_1/dense_3/BiasAdd:output:0*
T0*"
_output_shapes
:%(*
squeeze_dims
W
svdf_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_2/ExpandDims
ExpandDimssvdf_1/Squeeze:output:0svdf_2/ExpandDims/dim:output:0*
T0*&
_output_shapes
:%(�
'svdf_2/dense_4/Tensordot/ReadVariableOpReadVariableOp=svdf_2_dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel*
_output_shapes

:( *
dtype0w
&svdf_2/dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%   (   �
 svdf_2/dense_4/Tensordot/ReshapeReshapesvdf_2/ExpandDims:output:0/svdf_2/dense_4/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:%(�
svdf_2/dense_4/Tensordot/MatMulMatMul)svdf_2/dense_4/Tensordot/Reshape:output:0/svdf_2/dense_4/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:% w
svdf_2/dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %          �
svdf_2/dense_4/TensordotReshape)svdf_2/dense_4/Tensordot/MatMul:product:0'svdf_2/dense_4/Tensordot/shape:output:0*
T0*&
_output_shapes
:% �
svdf_2/stream_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_2/stream_2/PadPad!svdf_2/dense_4/Tensordot:output:0%svdf_2/stream_2/Pad/paddings:output:0*
T0*&
_output_shapes
:% �
;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp\svdf_2_stream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
2svdf_2/stream_2/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
:svdf_2/stream_2/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_2/stream_2/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativesvdf_2/stream_2/Pad:output:0Csvdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpNsvdf_2_stream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias*
_output_shapes
: *
dtype0�
*svdf_2/stream_2/depthwise_conv2d_2/BiasAddBiasAdd5svdf_2/stream_2/depthwise_conv2d_2/depthwise:output:0Asvdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: y
svdf_2/ReluRelu3svdf_2/stream_2/depthwise_conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
: �
'svdf_2/dense_5/Tensordot/ReadVariableOpReadVariableOp=svdf_2_dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel*
_output_shapes

: @*
dtype0w
&svdf_2/dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
 svdf_2/dense_5/Tensordot/ReshapeReshapesvdf_2/Relu:activations:0/svdf_2/dense_5/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
svdf_2/dense_5/Tensordot/MatMulMatMul)svdf_2/dense_5/Tensordot/Reshape:output:0/svdf_2/dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@w
svdf_2/dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
svdf_2/dense_5/TensordotReshape)svdf_2/dense_5/Tensordot/MatMul:product:0'svdf_2/dense_5/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
%svdf_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp9svdf_2_dense_5_biasadd_readvariableop_svdf_2_dense_5_bias*
_output_shapes
:@*
dtype0�
svdf_2/dense_5/BiasAddBiasAdd!svdf_2/dense_5/Tensordot:output:0-svdf_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@~
svdf_2/SqueezeSqueezesvdf_2/dense_5/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
W
svdf_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_3/ExpandDims
ExpandDimssvdf_2/Squeeze:output:0svdf_3/ExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
'svdf_3/dense_6/Tensordot/ReadVariableOpReadVariableOp=svdf_3_dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel*
_output_shapes

:@ *
dtype0w
&svdf_3/dense_6/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
 svdf_3/dense_6/Tensordot/ReshapeReshapesvdf_3/ExpandDims:output:0/svdf_3/dense_6/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
svdf_3/dense_6/Tensordot/MatMulMatMul)svdf_3/dense_6/Tensordot/Reshape:output:0/svdf_3/dense_6/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

: w
svdf_3/dense_6/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
svdf_3/dense_6/TensordotReshape)svdf_3/dense_6/Tensordot/MatMul:product:0'svdf_3/dense_6/Tensordot/shape:output:0*
T0*&
_output_shapes
: �
svdf_3/stream_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_3/stream_3/PadPad!svdf_3/dense_6/Tensordot:output:0%svdf_3/stream_3/Pad/paddings:output:0*
T0*&
_output_shapes
: �
;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp\svdf_3_stream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
2svdf_3/stream_3/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
:svdf_3/stream_3/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_3/stream_3/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativesvdf_3/stream_3/Pad:output:0Csvdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpNsvdf_3_stream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias*
_output_shapes
: *
dtype0�
*svdf_3/stream_3/depthwise_conv2d_3/BiasAddBiasAdd5svdf_3/stream_3/depthwise_conv2d_3/depthwise:output:0Asvdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: y
svdf_3/ReluRelu3svdf_3/stream_3/depthwise_conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
: �
'svdf_3/dense_7/Tensordot/ReadVariableOpReadVariableOp=svdf_3_dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel*
_output_shapes

: @*
dtype0w
&svdf_3/dense_7/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
 svdf_3/dense_7/Tensordot/ReshapeReshapesvdf_3/Relu:activations:0/svdf_3/dense_7/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
svdf_3/dense_7/Tensordot/MatMulMatMul)svdf_3/dense_7/Tensordot/Reshape:output:0/svdf_3/dense_7/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@w
svdf_3/dense_7/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
svdf_3/dense_7/TensordotReshape)svdf_3/dense_7/Tensordot/MatMul:product:0'svdf_3/dense_7/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
%svdf_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp9svdf_3_dense_7_biasadd_readvariableop_svdf_3_dense_7_bias*
_output_shapes
:@*
dtype0�
svdf_3/dense_7/BiasAddBiasAdd!svdf_3/dense_7/Tensordot:output:0-svdf_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@~
svdf_3/SqueezeSqueezesvdf_3/dense_7/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
W
svdf_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_4/ExpandDims
ExpandDimssvdf_3/Squeeze:output:0svdf_4/ExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
'svdf_4/dense_8/Tensordot/ReadVariableOpReadVariableOp=svdf_4_dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel*
_output_shapes

:@@*
dtype0w
&svdf_4/dense_8/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
 svdf_4/dense_8/Tensordot/ReshapeReshapesvdf_4/ExpandDims:output:0/svdf_4/dense_8/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
svdf_4/dense_8/Tensordot/MatMulMatMul)svdf_4/dense_8/Tensordot/Reshape:output:0/svdf_4/dense_8/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@w
svdf_4/dense_8/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
svdf_4/dense_8/TensordotReshape)svdf_4/dense_8/Tensordot/MatMul:product:0'svdf_4/dense_8/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
svdf_4/stream_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_4/stream_4/PadPad!svdf_4/dense_8/Tensordot:output:0%svdf_4/stream_4/Pad/paddings:output:0*
T0*&
_output_shapes
:@�
;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOp\svdf_4_stream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel*&
_output_shapes
:
@*
dtype0�
2svdf_4/stream_4/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      @      �
:svdf_4/stream_4/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_4/stream_4/depthwise_conv2d_4/depthwiseDepthwiseConv2dNativesvdf_4/stream_4/Pad:output:0Csvdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@*
paddingVALID*
strides
�
9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpNsvdf_4_stream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias*
_output_shapes
:@*
dtype0�
*svdf_4/stream_4/depthwise_conv2d_4/BiasAddBiasAdd5svdf_4/stream_4/depthwise_conv2d_4/depthwise:output:0Asvdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@y
svdf_4/ReluRelu3svdf_4/stream_4/depthwise_conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
:
@�
'svdf_4/dense_9/Tensordot/ReadVariableOpReadVariableOp=svdf_4_dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel*
_output_shapes

:@@*
dtype0w
&svdf_4/dense_9/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
 svdf_4/dense_9/Tensordot/ReshapeReshapesvdf_4/Relu:activations:0/svdf_4/dense_9/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
svdf_4/dense_9/Tensordot/MatMulMatMul)svdf_4/dense_9/Tensordot/Reshape:output:0/svdf_4/dense_9/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:
@w
svdf_4/dense_9/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      @   �
svdf_4/dense_9/TensordotReshape)svdf_4/dense_9/Tensordot/MatMul:product:0'svdf_4/dense_9/Tensordot/shape:output:0*
T0*&
_output_shapes
:
@�
%svdf_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp9svdf_4_dense_9_biasadd_readvariableop_svdf_4_dense_9_bias*
_output_shapes
:@*
dtype0�
svdf_4/dense_9/BiasAddBiasAdd!svdf_4/dense_9/Tensordot:output:0-svdf_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@~
svdf_4/SqueezeSqueezesvdf_4/dense_9/BiasAdd:output:0*
T0*"
_output_shapes
:
@*
squeeze_dims
W
svdf_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
svdf_5/ExpandDims
ExpandDimssvdf_4/Squeeze:output:0svdf_5/ExpandDims/dim:output:0*
T0*&
_output_shapes
:
@�
(svdf_5/dense_10/Tensordot/ReadVariableOpReadVariableOp?svdf_5_dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel*
_output_shapes
:	@�*
dtype0x
'svdf_5/dense_10/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
!svdf_5/dense_10/Tensordot/ReshapeReshapesvdf_5/ExpandDims:output:00svdf_5/dense_10/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
 svdf_5/dense_10/Tensordot/MatMulMatMul*svdf_5/dense_10/Tensordot/Reshape:output:00svdf_5/dense_10/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	
�x
svdf_5/dense_10/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      �   �
svdf_5/dense_10/TensordotReshape*svdf_5/dense_10/Tensordot/MatMul:product:0(svdf_5/dense_10/Tensordot/shape:output:0*
T0*'
_output_shapes
:
��
svdf_5/stream_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
svdf_5/stream_5/PadPad"svdf_5/dense_10/Tensordot:output:0%svdf_5/stream_5/Pad/paddings:output:0*
T0*'
_output_shapes
:
��
;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOp\svdf_5_stream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel*'
_output_shapes
:
�*
dtype0�
2svdf_5/stream_5/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      �      �
:svdf_5/stream_5/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
,svdf_5/stream_5/depthwise_conv2d_5/depthwiseDepthwiseConv2dNativesvdf_5/stream_5/Pad:output:0Csvdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpReadVariableOpNsvdf_5_stream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias*
_output_shapes	
:�*
dtype0�
*svdf_5/stream_5/depthwise_conv2d_5/BiasAddBiasAdd5svdf_5/stream_5/depthwise_conv2d_5/depthwise:output:0Asvdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�z
svdf_5/ReluRelu3svdf_5/stream_5/depthwise_conv2d_5/BiasAdd:output:0*
T0*'
_output_shapes
:�y
svdf_5/SqueezeSqueezesvdf_5/Relu:activations:0*
T0*#
_output_shapes
:�*
squeeze_dims
g
stream_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
stream_6/flatten/ReshapeReshapesvdf_5/Squeeze:output:0stream_6/flatten/Const:output:0*
T0*
_output_shapes
:	�i
dropout/IdentityIdentity!stream_6/flatten/Reshape:output:0*
T0*
_output_shapes
:	��
dense_11/MatMul/ReadVariableOpReadVariableOp.dense_11_matmul_readvariableop_dense_11_kernel*
_output_shapes
:	�*
dtype0�
dense_11/MatMulMatMuldropout/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
dense_11/BiasAdd/ReadVariableOpReadVariableOp-dense_11_biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:_
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp&^svdf_0/dense/Tensordot/ReadVariableOp&^svdf_0/dense_1/BiasAdd/ReadVariableOp(^svdf_0/dense_1/Tensordot/ReadVariableOp6^svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp8^svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp(^svdf_1/dense_2/Tensordot/ReadVariableOp&^svdf_1/dense_3/BiasAdd/ReadVariableOp(^svdf_1/dense_3/Tensordot/ReadVariableOp:^svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp<^svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp(^svdf_2/dense_4/Tensordot/ReadVariableOp&^svdf_2/dense_5/BiasAdd/ReadVariableOp(^svdf_2/dense_5/Tensordot/ReadVariableOp:^svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp<^svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp(^svdf_3/dense_6/Tensordot/ReadVariableOp&^svdf_3/dense_7/BiasAdd/ReadVariableOp(^svdf_3/dense_7/Tensordot/ReadVariableOp:^svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp<^svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp(^svdf_4/dense_8/Tensordot/ReadVariableOp&^svdf_4/dense_9/BiasAdd/ReadVariableOp(^svdf_4/dense_9/Tensordot/ReadVariableOp:^svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp<^svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp)^svdf_5/dense_10/Tensordot/ReadVariableOp:^svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp<^svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2N
%svdf_0/dense/Tensordot/ReadVariableOp%svdf_0/dense/Tensordot/ReadVariableOp2N
%svdf_0/dense_1/BiasAdd/ReadVariableOp%svdf_0/dense_1/BiasAdd/ReadVariableOp2R
'svdf_0/dense_1/Tensordot/ReadVariableOp'svdf_0/dense_1/Tensordot/ReadVariableOp2n
5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp5svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp2r
7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp7svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp2R
'svdf_1/dense_2/Tensordot/ReadVariableOp'svdf_1/dense_2/Tensordot/ReadVariableOp2N
%svdf_1/dense_3/BiasAdd/ReadVariableOp%svdf_1/dense_3/BiasAdd/ReadVariableOp2R
'svdf_1/dense_3/Tensordot/ReadVariableOp'svdf_1/dense_3/Tensordot/ReadVariableOp2v
9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp9svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2z
;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp;svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp2R
'svdf_2/dense_4/Tensordot/ReadVariableOp'svdf_2/dense_4/Tensordot/ReadVariableOp2N
%svdf_2/dense_5/BiasAdd/ReadVariableOp%svdf_2/dense_5/BiasAdd/ReadVariableOp2R
'svdf_2/dense_5/Tensordot/ReadVariableOp'svdf_2/dense_5/Tensordot/ReadVariableOp2v
9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp9svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2z
;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp;svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp2R
'svdf_3/dense_6/Tensordot/ReadVariableOp'svdf_3/dense_6/Tensordot/ReadVariableOp2N
%svdf_3/dense_7/BiasAdd/ReadVariableOp%svdf_3/dense_7/BiasAdd/ReadVariableOp2R
'svdf_3/dense_7/Tensordot/ReadVariableOp'svdf_3/dense_7/Tensordot/ReadVariableOp2v
9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp9svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2z
;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp;svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp2R
'svdf_4/dense_8/Tensordot/ReadVariableOp'svdf_4/dense_8/Tensordot/ReadVariableOp2N
%svdf_4/dense_9/BiasAdd/ReadVariableOp%svdf_4/dense_9/BiasAdd/ReadVariableOp2R
'svdf_4/dense_9/Tensordot/ReadVariableOp'svdf_4/dense_9/Tensordot/ReadVariableOp2v
9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp9svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2z
;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp;svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp2T
(svdf_5/dense_10/Tensordot/ReadVariableOp(svdf_5/dense_10/Tensordot/ReadVariableOp2v
9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp9svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2z
;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp;svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�
�
%__inference_svdf_5_layer_call_fn_4755

inputs)
svdf_5_dense_10_kernel:	@�;
 svdf_5_stream_5_depthwise_kernel:
�#
svdf_5_stream_5_bias:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_5_dense_10_kernel svdf_5_stream_5_depthwise_kernelsvdf_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:�*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_5_layer_call_and_return_conditional_losses_2537k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:
@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
@
 
_user_specified_nameinputs
�*
�
@__inference_svdf_4_layer_call_and_return_conditional_losses_2505

inputsH
6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel:@@o
Ustream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel:
@U
Gstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias:@H
6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel:@@@
2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias:@
identity�� dense_8/Tensordot/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp� dense_9/Tensordot/ReadVariableOp�2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp�4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_8/Tensordot/ReadVariableOpReadVariableOp6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel*
_output_shapes

:@@*
dtype0p
dense_8/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_8/Tensordot/ReshapeReshapeExpandDims:output:0(dense_8/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_8/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0 dense_8/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
stream_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_4/PadPaddense_8/Tensordot:output:0stream_4/Pad/paddings:output:0*
T0*&
_output_shapes
:@�
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOpUstream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel*&
_output_shapes
:
@*
dtype0�
+stream_4/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      @      �
3stream_4/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_4/depthwise_conv2d_4/depthwiseDepthwiseConv2dNativestream_4/Pad:output:0<stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@*
paddingVALID*
strides
�
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpGstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias*
_output_shapes
:@*
dtype0�
#stream_4/depthwise_conv2d_4/BiasAddBiasAdd.stream_4/depthwise_conv2d_4/depthwise:output:0:stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@k
ReluRelu,stream_4/depthwise_conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
:
@�
 dense_9/Tensordot/ReadVariableOpReadVariableOp6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel*
_output_shapes

:@@*
dtype0p
dense_9/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_9/Tensordot/ReshapeReshapeRelu:activations:0(dense_9/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:
@p
dense_9/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      @   �
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0 dense_9/Tensordot/shape:output:0*
T0*&
_output_shapes
:
@�
dense_9/BiasAdd/ReadVariableOpReadVariableOp2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias*
_output_shapes
:@*
dtype0�
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@p
SqueezeSqueezedense_9/BiasAdd:output:0*
T0*"
_output_shapes
:
@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:
@�
NoOpNoOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp3^stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp5^stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:@: : : : : 2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2h
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2l
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�*
�
@__inference_svdf_2_layer_call_and_return_conditional_losses_3053

inputsH
6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel:( o
Ustream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel:
 U
Gstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias: H
6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel: @@
2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias:@
identity�� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp�4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:%(�
 dense_4/Tensordot/ReadVariableOpReadVariableOp6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel*
_output_shapes

:( *
dtype0p
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%   (   �
dense_4/Tensordot/ReshapeReshapeExpandDims:output:0(dense_4/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:%(�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:% p
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %          �
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*&
_output_shapes
:% �
stream_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_2/PadPaddense_4/Tensordot:output:0stream_2/Pad/paddings:output:0*
T0*&
_output_shapes
:% �
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpUstream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_2/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_2/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_2/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativestream_2/Pad:output:0<stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias*
_output_shapes
: *
dtype0�
#stream_2/depthwise_conv2d_2/BiasAddBiasAdd.stream_2/depthwise_conv2d_2/depthwise:output:0:stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_2/depthwise_conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_5/Tensordot/ReadVariableOpReadVariableOp6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel*
_output_shapes

: @*
dtype0p
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_5/Tensordot/ReshapeReshapeRelu:activations:0(dense_5/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_5/BiasAdd/ReadVariableOpReadVariableOp2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias*
_output_shapes
:@*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_5/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp3^stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp5^stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:%(: : : : : 2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2h
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2l
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:J F
"
_output_shapes
:%(
 
_user_specified_nameinputs
�9
�
?__inference_model_layer_call_and_return_conditional_losses_3514

inputs,
svdf_0_svdf_0_dense_kernel:(?
%svdf_0_svdf_0_stream_depthwise_kernel:'
svdf_0_svdf_0_stream_bias:.
svdf_0_svdf_0_dense_1_kernel:((
svdf_0_svdf_0_dense_1_bias:(.
svdf_1_svdf_1_dense_2_kernel:( A
'svdf_1_svdf_1_stream_1_depthwise_kernel:
 )
svdf_1_svdf_1_stream_1_bias: .
svdf_1_svdf_1_dense_3_kernel: ((
svdf_1_svdf_1_dense_3_bias:(.
svdf_2_svdf_2_dense_4_kernel:( A
'svdf_2_svdf_2_stream_2_depthwise_kernel:
 )
svdf_2_svdf_2_stream_2_bias: .
svdf_2_svdf_2_dense_5_kernel: @(
svdf_2_svdf_2_dense_5_bias:@.
svdf_3_svdf_3_dense_6_kernel:@ A
'svdf_3_svdf_3_stream_3_depthwise_kernel:
 )
svdf_3_svdf_3_stream_3_bias: .
svdf_3_svdf_3_dense_7_kernel: @(
svdf_3_svdf_3_dense_7_bias:@.
svdf_4_svdf_4_dense_8_kernel:@@A
'svdf_4_svdf_4_stream_4_depthwise_kernel:
@)
svdf_4_svdf_4_stream_4_bias:@.
svdf_4_svdf_4_dense_9_kernel:@@(
svdf_4_svdf_4_dense_9_bias:@0
svdf_5_svdf_5_dense_10_kernel:	@�B
'svdf_5_svdf_5_stream_5_depthwise_kernel:
�*
svdf_5_svdf_5_stream_5_bias:	�+
dense_11_dense_11_kernel:	�$
dense_11_dense_11_bias:
identity�� dense_11/StatefulPartitionedCall�svdf_0/StatefulPartitionedCall�svdf_1/StatefulPartitionedCall�svdf_2/StatefulPartitionedCall�svdf_3/StatefulPartitionedCall�svdf_4/StatefulPartitionedCall�svdf_5/StatefulPartitionedCall�
speech_features/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_speech_features_layer_call_and_return_conditional_losses_3404�
svdf_0/StatefulPartitionedCallStatefulPartitionedCall(speech_features/PartitionedCall:output:0svdf_0_svdf_0_dense_kernel%svdf_0_svdf_0_stream_depthwise_kernelsvdf_0_svdf_0_stream_biassvdf_0_svdf_0_dense_1_kernelsvdf_0_svdf_0_dense_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:.(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_0_layer_call_and_return_conditional_losses_3305�
svdf_1/StatefulPartitionedCallStatefulPartitionedCall'svdf_0/StatefulPartitionedCall:output:0svdf_1_svdf_1_dense_2_kernel'svdf_1_svdf_1_stream_1_depthwise_kernelsvdf_1_svdf_1_stream_1_biassvdf_1_svdf_1_dense_3_kernelsvdf_1_svdf_1_dense_3_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:%(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_1_layer_call_and_return_conditional_losses_3179�
svdf_2/StatefulPartitionedCallStatefulPartitionedCall'svdf_1/StatefulPartitionedCall:output:0svdf_2_svdf_2_dense_4_kernel'svdf_2_svdf_2_stream_2_depthwise_kernelsvdf_2_svdf_2_stream_2_biassvdf_2_svdf_2_dense_5_kernelsvdf_2_svdf_2_dense_5_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_2_layer_call_and_return_conditional_losses_3053�
svdf_3/StatefulPartitionedCallStatefulPartitionedCall'svdf_2/StatefulPartitionedCall:output:0svdf_3_svdf_3_dense_6_kernel'svdf_3_svdf_3_stream_3_depthwise_kernelsvdf_3_svdf_3_stream_3_biassvdf_3_svdf_3_dense_7_kernelsvdf_3_svdf_3_dense_7_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_3_layer_call_and_return_conditional_losses_2927�
svdf_4/StatefulPartitionedCallStatefulPartitionedCall'svdf_3/StatefulPartitionedCall:output:0svdf_4_svdf_4_dense_8_kernel'svdf_4_svdf_4_stream_4_depthwise_kernelsvdf_4_svdf_4_stream_4_biassvdf_4_svdf_4_dense_9_kernelsvdf_4_svdf_4_dense_9_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:
@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_4_layer_call_and_return_conditional_losses_2801�
svdf_5/StatefulPartitionedCallStatefulPartitionedCall'svdf_4/StatefulPartitionedCall:output:0svdf_5_svdf_5_dense_10_kernel'svdf_5_svdf_5_stream_5_depthwise_kernelsvdf_5_svdf_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:�*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_5_layer_call_and_return_conditional_losses_2697�
stream_6/PartitionedCallPartitionedCall'svdf_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_2548�
dropout/PartitionedCallPartitionedCall!stream_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2636�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_2567o
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp!^dense_11/StatefulPartitionedCall^svdf_0/StatefulPartitionedCall^svdf_1/StatefulPartitionedCall^svdf_2/StatefulPartitionedCall^svdf_3/StatefulPartitionedCall^svdf_4/StatefulPartitionedCall^svdf_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
svdf_0/StatefulPartitionedCallsvdf_0/StatefulPartitionedCall2@
svdf_1/StatefulPartitionedCallsvdf_1/StatefulPartitionedCall2@
svdf_2/StatefulPartitionedCallsvdf_2/StatefulPartitionedCall2@
svdf_3/StatefulPartitionedCallsvdf_3/StatefulPartitionedCall2@
svdf_4/StatefulPartitionedCallsvdf_4/StatefulPartitionedCall2@
svdf_5/StatefulPartitionedCallsvdf_5/StatefulPartitionedCall:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�*
�
@__inference_svdf_4_layer_call_and_return_conditional_losses_4712

inputsH
6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel:@@o
Ustream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel:
@U
Gstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias:@H
6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel:@@@
2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias:@
identity�� dense_8/Tensordot/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp� dense_9/Tensordot/ReadVariableOp�2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp�4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_8/Tensordot/ReadVariableOpReadVariableOp6dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel*
_output_shapes

:@@*
dtype0p
dense_8/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_8/Tensordot/ReshapeReshapeExpandDims:output:0(dense_8/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_8/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0 dense_8/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
stream_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_4/PadPaddense_8/Tensordot:output:0stream_4/Pad/paddings:output:0*
T0*&
_output_shapes
:@�
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOpUstream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel*&
_output_shapes
:
@*
dtype0�
+stream_4/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      @      �
3stream_4/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_4/depthwise_conv2d_4/depthwiseDepthwiseConv2dNativestream_4/Pad:output:0<stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@*
paddingVALID*
strides
�
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpGstream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias*
_output_shapes
:@*
dtype0�
#stream_4/depthwise_conv2d_4/BiasAddBiasAdd.stream_4/depthwise_conv2d_4/depthwise:output:0:stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@k
ReluRelu,stream_4/depthwise_conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
:
@�
 dense_9/Tensordot/ReadVariableOpReadVariableOp6dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel*
_output_shapes

:@@*
dtype0p
dense_9/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_9/Tensordot/ReshapeReshapeRelu:activations:0(dense_9/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:
@p
dense_9/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      @   �
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0 dense_9/Tensordot/shape:output:0*
T0*&
_output_shapes
:
@�
dense_9/BiasAdd/ReadVariableOpReadVariableOp2dense_9_biasadd_readvariableop_svdf_4_dense_9_bias*
_output_shapes
:@*
dtype0�
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@p
SqueezeSqueezedense_9/BiasAdd:output:0*
T0*"
_output_shapes
:
@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:
@�
NoOpNoOp!^dense_8/Tensordot/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp3^stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp5^stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2h
2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2l
4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp4stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�
C
'__inference_stream_6_layer_call_fn_4818

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_2548X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*"
_input_shapes
:�:K G
#
_output_shapes
:�
 
_user_specified_nameinputs
�*
�
@__inference_svdf_1_layer_call_and_return_conditional_losses_4477

inputsH
6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel:( o
Ustream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel:
 U
Gstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias: H
6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel: (@
2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias:(
identity�� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp�4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:.(�
 dense_2/Tensordot/ReadVariableOpReadVariableOp6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel*
_output_shapes

:( *
dtype0p
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".   (   �
dense_2/Tensordot/ReshapeReshapeExpandDims:output:0(dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.(�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:. p
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .          �
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*&
_output_shapes
:. �
stream_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_1/PadPaddense_2/Tensordot:output:0stream_1/Pad/paddings:output:0*
T0*&
_output_shapes
:. �
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpUstream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativestream_1/Pad:output:0<stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:% *
paddingVALID*
strides
�
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias*
_output_shapes
: *
dtype0�
#stream_1/depthwise_conv2d_1/BiasAddBiasAdd.stream_1/depthwise_conv2d_1/depthwise:output:0:stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:% k
ReluRelu,stream_1/depthwise_conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:% �
 dense_3/Tensordot/ReadVariableOpReadVariableOp6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel*
_output_shapes

: (*
dtype0p
dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%       �
dense_3/Tensordot/ReshapeReshapeRelu:activations:0(dense_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:% �
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:%(p
dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %      (   �
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0 dense_3/Tensordot/shape:output:0*
T0*&
_output_shapes
:%(�
dense_3/BiasAdd/ReadVariableOpReadVariableOp2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias*
_output_shapes
:(*
dtype0�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:%(p
SqueezeSqueezedense_3/BiasAdd:output:0*
T0*"
_output_shapes
:%(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:%(�
NoOpNoOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp3^stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp5^stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:.(: : : : : 2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2h
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2l
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:J F
"
_output_shapes
:.(
 
_user_specified_nameinputs
�9
�
?__inference_model_layer_call_and_return_conditional_losses_3758
input_1,
svdf_0_svdf_0_dense_kernel:(?
%svdf_0_svdf_0_stream_depthwise_kernel:'
svdf_0_svdf_0_stream_bias:.
svdf_0_svdf_0_dense_1_kernel:((
svdf_0_svdf_0_dense_1_bias:(.
svdf_1_svdf_1_dense_2_kernel:( A
'svdf_1_svdf_1_stream_1_depthwise_kernel:
 )
svdf_1_svdf_1_stream_1_bias: .
svdf_1_svdf_1_dense_3_kernel: ((
svdf_1_svdf_1_dense_3_bias:(.
svdf_2_svdf_2_dense_4_kernel:( A
'svdf_2_svdf_2_stream_2_depthwise_kernel:
 )
svdf_2_svdf_2_stream_2_bias: .
svdf_2_svdf_2_dense_5_kernel: @(
svdf_2_svdf_2_dense_5_bias:@.
svdf_3_svdf_3_dense_6_kernel:@ A
'svdf_3_svdf_3_stream_3_depthwise_kernel:
 )
svdf_3_svdf_3_stream_3_bias: .
svdf_3_svdf_3_dense_7_kernel: @(
svdf_3_svdf_3_dense_7_bias:@.
svdf_4_svdf_4_dense_8_kernel:@@A
'svdf_4_svdf_4_stream_4_depthwise_kernel:
@)
svdf_4_svdf_4_stream_4_bias:@.
svdf_4_svdf_4_dense_9_kernel:@@(
svdf_4_svdf_4_dense_9_bias:@0
svdf_5_svdf_5_dense_10_kernel:	@�B
'svdf_5_svdf_5_stream_5_depthwise_kernel:
�*
svdf_5_svdf_5_stream_5_bias:	�+
dense_11_dense_11_kernel:	�$
dense_11_dense_11_bias:
identity�� dense_11/StatefulPartitionedCall�svdf_0/StatefulPartitionedCall�svdf_1/StatefulPartitionedCall�svdf_2/StatefulPartitionedCall�svdf_3/StatefulPartitionedCall�svdf_4/StatefulPartitionedCall�svdf_5/StatefulPartitionedCall�
speech_features/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_speech_features_layer_call_and_return_conditional_losses_3404�
svdf_0/StatefulPartitionedCallStatefulPartitionedCall(speech_features/PartitionedCall:output:0svdf_0_svdf_0_dense_kernel%svdf_0_svdf_0_stream_depthwise_kernelsvdf_0_svdf_0_stream_biassvdf_0_svdf_0_dense_1_kernelsvdf_0_svdf_0_dense_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:.(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_0_layer_call_and_return_conditional_losses_3305�
svdf_1/StatefulPartitionedCallStatefulPartitionedCall'svdf_0/StatefulPartitionedCall:output:0svdf_1_svdf_1_dense_2_kernel'svdf_1_svdf_1_stream_1_depthwise_kernelsvdf_1_svdf_1_stream_1_biassvdf_1_svdf_1_dense_3_kernelsvdf_1_svdf_1_dense_3_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:%(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_1_layer_call_and_return_conditional_losses_3179�
svdf_2/StatefulPartitionedCallStatefulPartitionedCall'svdf_1/StatefulPartitionedCall:output:0svdf_2_svdf_2_dense_4_kernel'svdf_2_svdf_2_stream_2_depthwise_kernelsvdf_2_svdf_2_stream_2_biassvdf_2_svdf_2_dense_5_kernelsvdf_2_svdf_2_dense_5_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_2_layer_call_and_return_conditional_losses_3053�
svdf_3/StatefulPartitionedCallStatefulPartitionedCall'svdf_2/StatefulPartitionedCall:output:0svdf_3_svdf_3_dense_6_kernel'svdf_3_svdf_3_stream_3_depthwise_kernelsvdf_3_svdf_3_stream_3_biassvdf_3_svdf_3_dense_7_kernelsvdf_3_svdf_3_dense_7_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_3_layer_call_and_return_conditional_losses_2927�
svdf_4/StatefulPartitionedCallStatefulPartitionedCall'svdf_3/StatefulPartitionedCall:output:0svdf_4_svdf_4_dense_8_kernel'svdf_4_svdf_4_stream_4_depthwise_kernelsvdf_4_svdf_4_stream_4_biassvdf_4_svdf_4_dense_9_kernelsvdf_4_svdf_4_dense_9_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:
@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_4_layer_call_and_return_conditional_losses_2801�
svdf_5/StatefulPartitionedCallStatefulPartitionedCall'svdf_4/StatefulPartitionedCall:output:0svdf_5_svdf_5_dense_10_kernel'svdf_5_svdf_5_stream_5_depthwise_kernelsvdf_5_svdf_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:�*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_5_layer_call_and_return_conditional_losses_2697�
stream_6/PartitionedCallPartitionedCall'svdf_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_2548�
dropout/PartitionedCallPartitionedCall!stream_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2636�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_2567o
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp!^dense_11/StatefulPartitionedCall^svdf_0/StatefulPartitionedCall^svdf_1/StatefulPartitionedCall^svdf_2/StatefulPartitionedCall^svdf_3/StatefulPartitionedCall^svdf_4/StatefulPartitionedCall^svdf_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
svdf_0/StatefulPartitionedCallsvdf_0/StatefulPartitionedCall2@
svdf_1/StatefulPartitionedCallsvdf_1/StatefulPartitionedCall2@
svdf_2/StatefulPartitionedCallsvdf_2/StatefulPartitionedCall2@
svdf_3/StatefulPartitionedCallsvdf_3/StatefulPartitionedCall2@
svdf_4/StatefulPartitionedCallsvdf_4/StatefulPartitionedCall2@
svdf_5/StatefulPartitionedCallsvdf_5/StatefulPartitionedCall:H D

_output_shapes
:	�}
!
_user_specified_name	input_1
�	
�
%__inference_svdf_1_layer_call_fn_4407

inputs'
svdf_1_dense_2_kernel:( :
 svdf_1_stream_1_depthwise_kernel:
 "
svdf_1_stream_1_bias: '
svdf_1_dense_3_kernel: (!
svdf_1_dense_3_bias:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_1_dense_2_kernel svdf_1_stream_1_depthwise_kernelsvdf_1_stream_1_biassvdf_1_dense_3_kernelsvdf_1_dense_3_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:%(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_1_layer_call_and_return_conditional_losses_3179j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:%(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:.(: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:.(
 
_user_specified_nameinputs
�	
�
%__inference_svdf_2_layer_call_fn_4497

inputs'
svdf_2_dense_4_kernel:( :
 svdf_2_stream_2_depthwise_kernel:
 "
svdf_2_stream_2_bias: '
svdf_2_dense_5_kernel: @!
svdf_2_dense_5_bias:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_2_dense_4_kernel svdf_2_stream_2_depthwise_kernelsvdf_2_stream_2_biassvdf_2_dense_5_kernelsvdf_2_dense_5_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_2_layer_call_and_return_conditional_losses_3053j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:%(: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:%(
 
_user_specified_nameinputs
�
�
%__inference_svdf_5_layer_call_fn_4763

inputs)
svdf_5_dense_10_kernel:	@�;
 svdf_5_stream_5_depthwise_kernel:
�#
svdf_5_stream_5_bias:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_5_dense_10_kernel svdf_5_stream_5_depthwise_kernelsvdf_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:�*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_5_layer_call_and_return_conditional_losses_2697k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:
@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
@
 
_user_specified_nameinputs
�	
�
%__inference_svdf_3_layer_call_fn_4587

inputs'
svdf_3_dense_6_kernel:@ :
 svdf_3_stream_3_depthwise_kernel:
 "
svdf_3_stream_3_bias: '
svdf_3_dense_7_kernel: @!
svdf_3_dense_7_bias:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_3_dense_6_kernel svdf_3_stream_3_depthwise_kernelsvdf_3_stream_3_biassvdf_3_dense_7_kernelsvdf_3_dense_7_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_3_layer_call_and_return_conditional_losses_2927j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�*
�
@__inference_svdf_3_layer_call_and_return_conditional_losses_2463

inputsH
6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel:@ o
Ustream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel:
 U
Gstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias: H
6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel: @@
2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias:@
identity�� dense_6/Tensordot/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp� dense_7/Tensordot/ReadVariableOp�2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp�4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_6/Tensordot/ReadVariableOpReadVariableOp6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel*
_output_shapes

:@ *
dtype0p
dense_6/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_6/Tensordot/ReshapeReshapeExpandDims:output:0(dense_6/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

: p
dense_6/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0 dense_6/Tensordot/shape:output:0*
T0*&
_output_shapes
: �
stream_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_3/PadPaddense_6/Tensordot:output:0stream_3/Pad/paddings:output:0*
T0*&
_output_shapes
: �
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpUstream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_3/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_3/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_3/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativestream_3/Pad:output:0<stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpGstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias*
_output_shapes
: *
dtype0�
#stream_3/depthwise_conv2d_3/BiasAddBiasAdd.stream_3/depthwise_conv2d_3/depthwise:output:0:stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_3/depthwise_conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_7/Tensordot/ReadVariableOpReadVariableOp6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel*
_output_shapes

: @*
dtype0p
dense_7/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_7/Tensordot/ReshapeReshapeRelu:activations:0(dense_7/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_7/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0 dense_7/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_7/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp3^stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp5^stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:@: : : : : 2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2h
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2l
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�*
�
@__inference_svdf_2_layer_call_and_return_conditional_losses_4567

inputsH
6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel:( o
Ustream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel:
 U
Gstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias: H
6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel: @@
2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias:@
identity�� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp�4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:%(�
 dense_4/Tensordot/ReadVariableOpReadVariableOp6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel*
_output_shapes

:( *
dtype0p
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%   (   �
dense_4/Tensordot/ReshapeReshapeExpandDims:output:0(dense_4/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:%(�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:% p
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %          �
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*&
_output_shapes
:% �
stream_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_2/PadPaddense_4/Tensordot:output:0stream_2/Pad/paddings:output:0*
T0*&
_output_shapes
:% �
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpUstream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_2/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_2/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_2/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativestream_2/Pad:output:0<stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias*
_output_shapes
: *
dtype0�
#stream_2/depthwise_conv2d_2/BiasAddBiasAdd.stream_2/depthwise_conv2d_2/depthwise:output:0:stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_2/depthwise_conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_5/Tensordot/ReadVariableOpReadVariableOp6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel*
_output_shapes

: @*
dtype0p
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_5/Tensordot/ReshapeReshapeRelu:activations:0(dense_5/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_5/BiasAdd/ReadVariableOpReadVariableOp2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias*
_output_shapes
:@*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_5/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp3^stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp5^stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:%(: : : : : 2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2h
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2l
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:J F
"
_output_shapes
:%(
 
_user_specified_nameinputs
�
_
A__inference_dropout_layer_call_and_return_conditional_losses_4839

inputs

identity_1F
IdentityIdentityinputs*
T0*
_output_shapes
:	�S

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	�"!

identity_1Identity_1:output:0*
_input_shapes
:	�:G C

_output_shapes
:	�
 
_user_specified_nameinputs
��
�#
__inference__wrapped_model_2280
input_1Q
?model_svdf_0_dense_tensordot_readvariableop_svdf_0_dense_kernel:(v
\model_svdf_0_stream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel:\
Nmodel_svdf_0_stream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias:U
Cmodel_svdf_0_dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel:(M
?model_svdf_0_dense_1_biasadd_readvariableop_svdf_0_dense_1_bias:(U
Cmodel_svdf_1_dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel:( |
bmodel_svdf_1_stream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel:
 b
Tmodel_svdf_1_stream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias: U
Cmodel_svdf_1_dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel: (M
?model_svdf_1_dense_3_biasadd_readvariableop_svdf_1_dense_3_bias:(U
Cmodel_svdf_2_dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel:( |
bmodel_svdf_2_stream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel:
 b
Tmodel_svdf_2_stream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias: U
Cmodel_svdf_2_dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel: @M
?model_svdf_2_dense_5_biasadd_readvariableop_svdf_2_dense_5_bias:@U
Cmodel_svdf_3_dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel:@ |
bmodel_svdf_3_stream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel:
 b
Tmodel_svdf_3_stream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias: U
Cmodel_svdf_3_dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel: @M
?model_svdf_3_dense_7_biasadd_readvariableop_svdf_3_dense_7_bias:@U
Cmodel_svdf_4_dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel:@@|
bmodel_svdf_4_stream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel:
@b
Tmodel_svdf_4_stream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias:@U
Cmodel_svdf_4_dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel:@@M
?model_svdf_4_dense_9_biasadd_readvariableop_svdf_4_dense_9_bias:@X
Emodel_svdf_5_dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel:	@�}
bmodel_svdf_5_stream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel:
�c
Tmodel_svdf_5_stream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias:	�G
4model_dense_11_matmul_readvariableop_dense_11_kernel:	�A
3model_dense_11_biasadd_readvariableop_dense_11_bias:
identity��%model/dense_11/BiasAdd/ReadVariableOp�$model/dense_11/MatMul/ReadVariableOp�+model/svdf_0/dense/Tensordot/ReadVariableOp�+model/svdf_0/dense_1/BiasAdd/ReadVariableOp�-model/svdf_0/dense_1/Tensordot/ReadVariableOp�;model/svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp�=model/svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp�-model/svdf_1/dense_2/Tensordot/ReadVariableOp�+model/svdf_1/dense_3/BiasAdd/ReadVariableOp�-model/svdf_1/dense_3/Tensordot/ReadVariableOp�?model/svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp�Amodel/svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp�-model/svdf_2/dense_4/Tensordot/ReadVariableOp�+model/svdf_2/dense_5/BiasAdd/ReadVariableOp�-model/svdf_2/dense_5/Tensordot/ReadVariableOp�?model/svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp�Amodel/svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp�-model/svdf_3/dense_6/Tensordot/ReadVariableOp�+model/svdf_3/dense_7/BiasAdd/ReadVariableOp�-model/svdf_3/dense_7/Tensordot/ReadVariableOp�?model/svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp�Amodel/svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp�-model/svdf_4/dense_8/Tensordot/ReadVariableOp�+model/svdf_4/dense_9/BiasAdd/ReadVariableOp�-model/svdf_4/dense_9/Tensordot/ReadVariableOp�?model/svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp�Amodel/svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp�.model/svdf_5/dense_10/Tensordot/ReadVariableOp�?model/svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp�Amodel/svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpu
$model/speech_features/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       �
model/speech_features/transpose	Transposeinput_1-model/speech_features/transpose/perm:output:0*
T0*
_output_shapes
:	�}�
&model/speech_features/AudioSpectrogramAudioSpectrogram#model/speech_features/transpose:y:0*#
_output_shapes
:1�*
magnitude_squared(*
stride�*
window_size�i
&model/speech_features/Mfcc/sample_rateConst*
_output_shapes
: *
dtype0*
value
B :�}�
model/speech_features/MfccMfcc4model/speech_features/AudioSpectrogram:spectrogram:0/model/speech_features/Mfcc/sample_rate:output:0*"
_output_shapes
:1(*
dct_coefficient_count(*
filterbank_channel_countP*
upper_frequency_limit% ��E�
&model/speech_features/normalizer/sub/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�                                                                                                                                                                �
$model/speech_features/normalizer/subSub#model/speech_features/Mfcc:output:0/model/speech_features/normalizer/sub/y:output:0*
T0*"
_output_shapes
:1(�
*model/speech_features/normalizer/truediv/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?�
(model/speech_features/normalizer/truedivRealDiv(model/speech_features/normalizer/sub:z:03model/speech_features/normalizer/truediv/y:output:0*
T0*"
_output_shapes
:1(]
model/svdf_0/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/svdf_0/ExpandDims
ExpandDims,model/speech_features/normalizer/truediv:z:0$model/svdf_0/ExpandDims/dim:output:0*
T0*&
_output_shapes
:1(�
+model/svdf_0/dense/Tensordot/ReadVariableOpReadVariableOp?model_svdf_0_dense_tensordot_readvariableop_svdf_0_dense_kernel*
_output_shapes

:(*
dtype0{
*model/svdf_0/dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"1   (   �
$model/svdf_0/dense/Tensordot/ReshapeReshape model/svdf_0/ExpandDims:output:03model/svdf_0/dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:1(�
#model/svdf_0/dense/Tensordot/MatMulMatMul-model/svdf_0/dense/Tensordot/Reshape:output:03model/svdf_0/dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:1{
"model/svdf_0/dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   1         �
model/svdf_0/dense/TensordotReshape-model/svdf_0/dense/Tensordot/MatMul:product:0+model/svdf_0/dense/Tensordot/shape:output:0*
T0*&
_output_shapes
:1�
 model/svdf_0/stream/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
model/svdf_0/stream/PadPad%model/svdf_0/dense/Tensordot:output:0)model/svdf_0/stream/Pad/paddings:output:0*
T0*&
_output_shapes
:1�
=model/svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOp\model_svdf_0_stream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel*&
_output_shapes
:*
dtype0�
4model/svdf_0/stream/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
<model/svdf_0/stream/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
.model/svdf_0/stream/depthwise_conv2d/depthwiseDepthwiseConv2dNative model/svdf_0/stream/Pad:output:0Emodel/svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:.*
paddingVALID*
strides
�
;model/svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpNmodel_svdf_0_stream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias*
_output_shapes
:*
dtype0�
,model/svdf_0/stream/depthwise_conv2d/BiasAddBiasAdd7model/svdf_0/stream/depthwise_conv2d/depthwise:output:0Cmodel/svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.�
model/svdf_0/ReluRelu5model/svdf_0/stream/depthwise_conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:.�
-model/svdf_0/dense_1/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_0_dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel*
_output_shapes

:(*
dtype0}
,model/svdf_0/dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".      �
&model/svdf_0/dense_1/Tensordot/ReshapeReshapemodel/svdf_0/Relu:activations:05model/svdf_0/dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.�
%model/svdf_0/dense_1/Tensordot/MatMulMatMul/model/svdf_0/dense_1/Tensordot/Reshape:output:05model/svdf_0/dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:.(}
$model/svdf_0/dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .      (   �
model/svdf_0/dense_1/TensordotReshape/model/svdf_0/dense_1/Tensordot/MatMul:product:0-model/svdf_0/dense_1/Tensordot/shape:output:0*
T0*&
_output_shapes
:.(�
+model/svdf_0/dense_1/BiasAdd/ReadVariableOpReadVariableOp?model_svdf_0_dense_1_biasadd_readvariableop_svdf_0_dense_1_bias*
_output_shapes
:(*
dtype0�
model/svdf_0/dense_1/BiasAddBiasAdd'model/svdf_0/dense_1/Tensordot:output:03model/svdf_0/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.(�
model/svdf_0/SqueezeSqueeze%model/svdf_0/dense_1/BiasAdd:output:0*
T0*"
_output_shapes
:.(*
squeeze_dims
]
model/svdf_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/svdf_1/ExpandDims
ExpandDimsmodel/svdf_0/Squeeze:output:0$model/svdf_1/ExpandDims/dim:output:0*
T0*&
_output_shapes
:.(�
-model/svdf_1/dense_2/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_1_dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel*
_output_shapes

:( *
dtype0}
,model/svdf_1/dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".   (   �
&model/svdf_1/dense_2/Tensordot/ReshapeReshape model/svdf_1/ExpandDims:output:05model/svdf_1/dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.(�
%model/svdf_1/dense_2/Tensordot/MatMulMatMul/model/svdf_1/dense_2/Tensordot/Reshape:output:05model/svdf_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:. }
$model/svdf_1/dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .          �
model/svdf_1/dense_2/TensordotReshape/model/svdf_1/dense_2/Tensordot/MatMul:product:0-model/svdf_1/dense_2/Tensordot/shape:output:0*
T0*&
_output_shapes
:. �
"model/svdf_1/stream_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
model/svdf_1/stream_1/PadPad'model/svdf_1/dense_2/Tensordot:output:0+model/svdf_1/stream_1/Pad/paddings:output:0*
T0*&
_output_shapes
:. �
Amodel/svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpbmodel_svdf_1_stream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
8model/svdf_1/stream_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
@model/svdf_1/stream_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
2model/svdf_1/stream_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNative"model/svdf_1/stream_1/Pad:output:0Imodel/svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:% *
paddingVALID*
strides
�
?model/svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpTmodel_svdf_1_stream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias*
_output_shapes
: *
dtype0�
0model/svdf_1/stream_1/depthwise_conv2d_1/BiasAddBiasAdd;model/svdf_1/stream_1/depthwise_conv2d_1/depthwise:output:0Gmodel/svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:% �
model/svdf_1/ReluRelu9model/svdf_1/stream_1/depthwise_conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:% �
-model/svdf_1/dense_3/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_1_dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel*
_output_shapes

: (*
dtype0}
,model/svdf_1/dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%       �
&model/svdf_1/dense_3/Tensordot/ReshapeReshapemodel/svdf_1/Relu:activations:05model/svdf_1/dense_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:% �
%model/svdf_1/dense_3/Tensordot/MatMulMatMul/model/svdf_1/dense_3/Tensordot/Reshape:output:05model/svdf_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:%(}
$model/svdf_1/dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %      (   �
model/svdf_1/dense_3/TensordotReshape/model/svdf_1/dense_3/Tensordot/MatMul:product:0-model/svdf_1/dense_3/Tensordot/shape:output:0*
T0*&
_output_shapes
:%(�
+model/svdf_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp?model_svdf_1_dense_3_biasadd_readvariableop_svdf_1_dense_3_bias*
_output_shapes
:(*
dtype0�
model/svdf_1/dense_3/BiasAddBiasAdd'model/svdf_1/dense_3/Tensordot:output:03model/svdf_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:%(�
model/svdf_1/SqueezeSqueeze%model/svdf_1/dense_3/BiasAdd:output:0*
T0*"
_output_shapes
:%(*
squeeze_dims
]
model/svdf_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/svdf_2/ExpandDims
ExpandDimsmodel/svdf_1/Squeeze:output:0$model/svdf_2/ExpandDims/dim:output:0*
T0*&
_output_shapes
:%(�
-model/svdf_2/dense_4/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_2_dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel*
_output_shapes

:( *
dtype0}
,model/svdf_2/dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%   (   �
&model/svdf_2/dense_4/Tensordot/ReshapeReshape model/svdf_2/ExpandDims:output:05model/svdf_2/dense_4/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:%(�
%model/svdf_2/dense_4/Tensordot/MatMulMatMul/model/svdf_2/dense_4/Tensordot/Reshape:output:05model/svdf_2/dense_4/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:% }
$model/svdf_2/dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %          �
model/svdf_2/dense_4/TensordotReshape/model/svdf_2/dense_4/Tensordot/MatMul:product:0-model/svdf_2/dense_4/Tensordot/shape:output:0*
T0*&
_output_shapes
:% �
"model/svdf_2/stream_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
model/svdf_2/stream_2/PadPad'model/svdf_2/dense_4/Tensordot:output:0+model/svdf_2/stream_2/Pad/paddings:output:0*
T0*&
_output_shapes
:% �
Amodel/svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpbmodel_svdf_2_stream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
8model/svdf_2/stream_2/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
@model/svdf_2/stream_2/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
2model/svdf_2/stream_2/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative"model/svdf_2/stream_2/Pad:output:0Imodel/svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
?model/svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpTmodel_svdf_2_stream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias*
_output_shapes
: *
dtype0�
0model/svdf_2/stream_2/depthwise_conv2d_2/BiasAddBiasAdd;model/svdf_2/stream_2/depthwise_conv2d_2/depthwise:output:0Gmodel/svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
model/svdf_2/ReluRelu9model/svdf_2/stream_2/depthwise_conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
: �
-model/svdf_2/dense_5/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_2_dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel*
_output_shapes

: @*
dtype0}
,model/svdf_2/dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
&model/svdf_2/dense_5/Tensordot/ReshapeReshapemodel/svdf_2/Relu:activations:05model/svdf_2/dense_5/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
%model/svdf_2/dense_5/Tensordot/MatMulMatMul/model/svdf_2/dense_5/Tensordot/Reshape:output:05model/svdf_2/dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@}
$model/svdf_2/dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
model/svdf_2/dense_5/TensordotReshape/model/svdf_2/dense_5/Tensordot/MatMul:product:0-model/svdf_2/dense_5/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
+model/svdf_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp?model_svdf_2_dense_5_biasadd_readvariableop_svdf_2_dense_5_bias*
_output_shapes
:@*
dtype0�
model/svdf_2/dense_5/BiasAddBiasAdd'model/svdf_2/dense_5/Tensordot:output:03model/svdf_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@�
model/svdf_2/SqueezeSqueeze%model/svdf_2/dense_5/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
]
model/svdf_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/svdf_3/ExpandDims
ExpandDimsmodel/svdf_2/Squeeze:output:0$model/svdf_3/ExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
-model/svdf_3/dense_6/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_3_dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel*
_output_shapes

:@ *
dtype0}
,model/svdf_3/dense_6/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
&model/svdf_3/dense_6/Tensordot/ReshapeReshape model/svdf_3/ExpandDims:output:05model/svdf_3/dense_6/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
%model/svdf_3/dense_6/Tensordot/MatMulMatMul/model/svdf_3/dense_6/Tensordot/Reshape:output:05model/svdf_3/dense_6/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

: }
$model/svdf_3/dense_6/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
model/svdf_3/dense_6/TensordotReshape/model/svdf_3/dense_6/Tensordot/MatMul:product:0-model/svdf_3/dense_6/Tensordot/shape:output:0*
T0*&
_output_shapes
: �
"model/svdf_3/stream_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
model/svdf_3/stream_3/PadPad'model/svdf_3/dense_6/Tensordot:output:0+model/svdf_3/stream_3/Pad/paddings:output:0*
T0*&
_output_shapes
: �
Amodel/svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpbmodel_svdf_3_stream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
8model/svdf_3/stream_3/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
@model/svdf_3/stream_3/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
2model/svdf_3/stream_3/depthwise_conv2d_3/depthwiseDepthwiseConv2dNative"model/svdf_3/stream_3/Pad:output:0Imodel/svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
?model/svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpTmodel_svdf_3_stream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias*
_output_shapes
: *
dtype0�
0model/svdf_3/stream_3/depthwise_conv2d_3/BiasAddBiasAdd;model/svdf_3/stream_3/depthwise_conv2d_3/depthwise:output:0Gmodel/svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: �
model/svdf_3/ReluRelu9model/svdf_3/stream_3/depthwise_conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
: �
-model/svdf_3/dense_7/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_3_dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel*
_output_shapes

: @*
dtype0}
,model/svdf_3/dense_7/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
&model/svdf_3/dense_7/Tensordot/ReshapeReshapemodel/svdf_3/Relu:activations:05model/svdf_3/dense_7/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
%model/svdf_3/dense_7/Tensordot/MatMulMatMul/model/svdf_3/dense_7/Tensordot/Reshape:output:05model/svdf_3/dense_7/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@}
$model/svdf_3/dense_7/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
model/svdf_3/dense_7/TensordotReshape/model/svdf_3/dense_7/Tensordot/MatMul:product:0-model/svdf_3/dense_7/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
+model/svdf_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp?model_svdf_3_dense_7_biasadd_readvariableop_svdf_3_dense_7_bias*
_output_shapes
:@*
dtype0�
model/svdf_3/dense_7/BiasAddBiasAdd'model/svdf_3/dense_7/Tensordot:output:03model/svdf_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@�
model/svdf_3/SqueezeSqueeze%model/svdf_3/dense_7/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
]
model/svdf_4/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/svdf_4/ExpandDims
ExpandDimsmodel/svdf_3/Squeeze:output:0$model/svdf_4/ExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
-model/svdf_4/dense_8/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_4_dense_8_tensordot_readvariableop_svdf_4_dense_8_kernel*
_output_shapes

:@@*
dtype0}
,model/svdf_4/dense_8/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
&model/svdf_4/dense_8/Tensordot/ReshapeReshape model/svdf_4/ExpandDims:output:05model/svdf_4/dense_8/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
%model/svdf_4/dense_8/Tensordot/MatMulMatMul/model/svdf_4/dense_8/Tensordot/Reshape:output:05model/svdf_4/dense_8/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@}
$model/svdf_4/dense_8/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
model/svdf_4/dense_8/TensordotReshape/model/svdf_4/dense_8/Tensordot/MatMul:product:0-model/svdf_4/dense_8/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
"model/svdf_4/stream_4/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
model/svdf_4/stream_4/PadPad'model/svdf_4/dense_8/Tensordot:output:0+model/svdf_4/stream_4/Pad/paddings:output:0*
T0*&
_output_shapes
:@�
Amodel/svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpReadVariableOpbmodel_svdf_4_stream_4_depthwise_conv2d_4_depthwise_readvariableop_svdf_4_stream_4_depthwise_kernel*&
_output_shapes
:
@*
dtype0�
8model/svdf_4/stream_4/depthwise_conv2d_4/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      @      �
@model/svdf_4/stream_4/depthwise_conv2d_4/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
2model/svdf_4/stream_4/depthwise_conv2d_4/depthwiseDepthwiseConv2dNative"model/svdf_4/stream_4/Pad:output:0Imodel/svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@*
paddingVALID*
strides
�
?model/svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpReadVariableOpTmodel_svdf_4_stream_4_depthwise_conv2d_4_biasadd_readvariableop_svdf_4_stream_4_bias*
_output_shapes
:@*
dtype0�
0model/svdf_4/stream_4/depthwise_conv2d_4/BiasAddBiasAdd;model/svdf_4/stream_4/depthwise_conv2d_4/depthwise:output:0Gmodel/svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@�
model/svdf_4/ReluRelu9model/svdf_4/stream_4/depthwise_conv2d_4/BiasAdd:output:0*
T0*&
_output_shapes
:
@�
-model/svdf_4/dense_9/Tensordot/ReadVariableOpReadVariableOpCmodel_svdf_4_dense_9_tensordot_readvariableop_svdf_4_dense_9_kernel*
_output_shapes

:@@*
dtype0}
,model/svdf_4/dense_9/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
&model/svdf_4/dense_9/Tensordot/ReshapeReshapemodel/svdf_4/Relu:activations:05model/svdf_4/dense_9/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
%model/svdf_4/dense_9/Tensordot/MatMulMatMul/model/svdf_4/dense_9/Tensordot/Reshape:output:05model/svdf_4/dense_9/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:
@}
$model/svdf_4/dense_9/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      @   �
model/svdf_4/dense_9/TensordotReshape/model/svdf_4/dense_9/Tensordot/MatMul:product:0-model/svdf_4/dense_9/Tensordot/shape:output:0*
T0*&
_output_shapes
:
@�
+model/svdf_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp?model_svdf_4_dense_9_biasadd_readvariableop_svdf_4_dense_9_bias*
_output_shapes
:@*
dtype0�
model/svdf_4/dense_9/BiasAddBiasAdd'model/svdf_4/dense_9/Tensordot:output:03model/svdf_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:
@�
model/svdf_4/SqueezeSqueeze%model/svdf_4/dense_9/BiasAdd:output:0*
T0*"
_output_shapes
:
@*
squeeze_dims
]
model/svdf_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model/svdf_5/ExpandDims
ExpandDimsmodel/svdf_4/Squeeze:output:0$model/svdf_5/ExpandDims/dim:output:0*
T0*&
_output_shapes
:
@�
.model/svdf_5/dense_10/Tensordot/ReadVariableOpReadVariableOpEmodel_svdf_5_dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel*
_output_shapes
:	@�*
dtype0~
-model/svdf_5/dense_10/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
'model/svdf_5/dense_10/Tensordot/ReshapeReshape model/svdf_5/ExpandDims:output:06model/svdf_5/dense_10/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
&model/svdf_5/dense_10/Tensordot/MatMulMatMul0model/svdf_5/dense_10/Tensordot/Reshape:output:06model/svdf_5/dense_10/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	
�~
%model/svdf_5/dense_10/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      �   �
model/svdf_5/dense_10/TensordotReshape0model/svdf_5/dense_10/Tensordot/MatMul:product:0.model/svdf_5/dense_10/Tensordot/shape:output:0*
T0*'
_output_shapes
:
��
"model/svdf_5/stream_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
model/svdf_5/stream_5/PadPad(model/svdf_5/dense_10/Tensordot:output:0+model/svdf_5/stream_5/Pad/paddings:output:0*
T0*'
_output_shapes
:
��
Amodel/svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOpbmodel_svdf_5_stream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel*'
_output_shapes
:
�*
dtype0�
8model/svdf_5/stream_5/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      �      �
@model/svdf_5/stream_5/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
2model/svdf_5/stream_5/depthwise_conv2d_5/depthwiseDepthwiseConv2dNative"model/svdf_5/stream_5/Pad:output:0Imodel/svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
?model/svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpReadVariableOpTmodel_svdf_5_stream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias*
_output_shapes	
:�*
dtype0�
0model/svdf_5/stream_5/depthwise_conv2d_5/BiasAddBiasAdd;model/svdf_5/stream_5/depthwise_conv2d_5/depthwise:output:0Gmodel/svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��
model/svdf_5/ReluRelu9model/svdf_5/stream_5/depthwise_conv2d_5/BiasAdd:output:0*
T0*'
_output_shapes
:��
model/svdf_5/SqueezeSqueezemodel/svdf_5/Relu:activations:0*
T0*#
_output_shapes
:�*
squeeze_dims
m
model/stream_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����   �
model/stream_6/flatten/ReshapeReshapemodel/svdf_5/Squeeze:output:0%model/stream_6/flatten/Const:output:0*
T0*
_output_shapes
:	�u
model/dropout/IdentityIdentity'model/stream_6/flatten/Reshape:output:0*
T0*
_output_shapes
:	��
$model/dense_11/MatMul/ReadVariableOpReadVariableOp4model_dense_11_matmul_readvariableop_dense_11_kernel*
_output_shapes
:	�*
dtype0�
model/dense_11/MatMulMatMulmodel/dropout/Identity:output:0,model/dense_11/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
%model/dense_11/BiasAdd/ReadVariableOpReadVariableOp3model_dense_11_biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0�
model/dense_11/BiasAddBiasAddmodel/dense_11/MatMul:product:0-model/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:e
IdentityIdentitymodel/dense_11/BiasAdd:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp&^model/dense_11/BiasAdd/ReadVariableOp%^model/dense_11/MatMul/ReadVariableOp,^model/svdf_0/dense/Tensordot/ReadVariableOp,^model/svdf_0/dense_1/BiasAdd/ReadVariableOp.^model/svdf_0/dense_1/Tensordot/ReadVariableOp<^model/svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp>^model/svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp.^model/svdf_1/dense_2/Tensordot/ReadVariableOp,^model/svdf_1/dense_3/BiasAdd/ReadVariableOp.^model/svdf_1/dense_3/Tensordot/ReadVariableOp@^model/svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpB^model/svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp.^model/svdf_2/dense_4/Tensordot/ReadVariableOp,^model/svdf_2/dense_5/BiasAdd/ReadVariableOp.^model/svdf_2/dense_5/Tensordot/ReadVariableOp@^model/svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpB^model/svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp.^model/svdf_3/dense_6/Tensordot/ReadVariableOp,^model/svdf_3/dense_7/BiasAdd/ReadVariableOp.^model/svdf_3/dense_7/Tensordot/ReadVariableOp@^model/svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpB^model/svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp.^model/svdf_4/dense_8/Tensordot/ReadVariableOp,^model/svdf_4/dense_9/BiasAdd/ReadVariableOp.^model/svdf_4/dense_9/Tensordot/ReadVariableOp@^model/svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOpB^model/svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp/^model/svdf_5/dense_10/Tensordot/ReadVariableOp@^model/svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpB^model/svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%model/dense_11/BiasAdd/ReadVariableOp%model/dense_11/BiasAdd/ReadVariableOp2L
$model/dense_11/MatMul/ReadVariableOp$model/dense_11/MatMul/ReadVariableOp2Z
+model/svdf_0/dense/Tensordot/ReadVariableOp+model/svdf_0/dense/Tensordot/ReadVariableOp2Z
+model/svdf_0/dense_1/BiasAdd/ReadVariableOp+model/svdf_0/dense_1/BiasAdd/ReadVariableOp2^
-model/svdf_0/dense_1/Tensordot/ReadVariableOp-model/svdf_0/dense_1/Tensordot/ReadVariableOp2z
;model/svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp;model/svdf_0/stream/depthwise_conv2d/BiasAdd/ReadVariableOp2~
=model/svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp=model/svdf_0/stream/depthwise_conv2d/depthwise/ReadVariableOp2^
-model/svdf_1/dense_2/Tensordot/ReadVariableOp-model/svdf_1/dense_2/Tensordot/ReadVariableOp2Z
+model/svdf_1/dense_3/BiasAdd/ReadVariableOp+model/svdf_1/dense_3/BiasAdd/ReadVariableOp2^
-model/svdf_1/dense_3/Tensordot/ReadVariableOp-model/svdf_1/dense_3/Tensordot/ReadVariableOp2�
?model/svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp?model/svdf_1/stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2�
Amodel/svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpAmodel/svdf_1/stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp2^
-model/svdf_2/dense_4/Tensordot/ReadVariableOp-model/svdf_2/dense_4/Tensordot/ReadVariableOp2Z
+model/svdf_2/dense_5/BiasAdd/ReadVariableOp+model/svdf_2/dense_5/BiasAdd/ReadVariableOp2^
-model/svdf_2/dense_5/Tensordot/ReadVariableOp-model/svdf_2/dense_5/Tensordot/ReadVariableOp2�
?model/svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp?model/svdf_2/stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2�
Amodel/svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpAmodel/svdf_2/stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp2^
-model/svdf_3/dense_6/Tensordot/ReadVariableOp-model/svdf_3/dense_6/Tensordot/ReadVariableOp2Z
+model/svdf_3/dense_7/BiasAdd/ReadVariableOp+model/svdf_3/dense_7/BiasAdd/ReadVariableOp2^
-model/svdf_3/dense_7/Tensordot/ReadVariableOp-model/svdf_3/dense_7/Tensordot/ReadVariableOp2�
?model/svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp?model/svdf_3/stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2�
Amodel/svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpAmodel/svdf_3/stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp2^
-model/svdf_4/dense_8/Tensordot/ReadVariableOp-model/svdf_4/dense_8/Tensordot/ReadVariableOp2Z
+model/svdf_4/dense_9/BiasAdd/ReadVariableOp+model/svdf_4/dense_9/BiasAdd/ReadVariableOp2^
-model/svdf_4/dense_9/Tensordot/ReadVariableOp-model/svdf_4/dense_9/Tensordot/ReadVariableOp2�
?model/svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp?model/svdf_4/stream_4/depthwise_conv2d_4/BiasAdd/ReadVariableOp2�
Amodel/svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOpAmodel/svdf_4/stream_4/depthwise_conv2d_4/depthwise/ReadVariableOp2`
.model/svdf_5/dense_10/Tensordot/ReadVariableOp.model/svdf_5/dense_10/Tensordot/ReadVariableOp2�
?model/svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp?model/svdf_5/stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2�
Amodel/svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpAmodel/svdf_5/stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:H D

_output_shapes
:	�}
!
_user_specified_name	input_1
�	
�
%__inference_svdf_1_layer_call_fn_4397

inputs'
svdf_1_dense_2_kernel:( :
 svdf_1_stream_1_depthwise_kernel:
 "
svdf_1_stream_1_bias: '
svdf_1_dense_3_kernel: (!
svdf_1_dense_3_bias:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_1_dense_2_kernel svdf_1_stream_1_depthwise_kernelsvdf_1_stream_1_biassvdf_1_dense_3_kernelsvdf_1_dense_3_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:%(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_1_layer_call_and_return_conditional_losses_2379j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:%(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:.(: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:.(
 
_user_specified_nameinputs
�)
�
@__inference_svdf_0_layer_call_and_return_conditional_losses_2337

inputsD
2dense_tensordot_readvariableop_svdf_0_dense_kernel:(i
Ostream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel:O
Astream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias:H
6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel:(@
2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias:(
identity��dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�.stream/depthwise_conv2d/BiasAdd/ReadVariableOp�0stream/depthwise_conv2d/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:1(�
dense/Tensordot/ReadVariableOpReadVariableOp2dense_tensordot_readvariableop_svdf_0_dense_kernel*
_output_shapes

:(*
dtype0n
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"1   (   �
dense/Tensordot/ReshapeReshapeExpandDims:output:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:1(�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:1n
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   1         �
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*&
_output_shapes
:1�
stream/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 z

stream/PadPaddense/Tensordot:output:0stream/Pad/paddings:output:0*
T0*&
_output_shapes
:1�
0stream/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpOstream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel*&
_output_shapes
:*
dtype0�
'stream/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
/stream/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
!stream/depthwise_conv2d/depthwiseDepthwiseConv2dNativestream/Pad:output:08stream/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:.*
paddingVALID*
strides
�
.stream/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpAstream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias*
_output_shapes
:*
dtype0�
stream/depthwise_conv2d/BiasAddBiasAdd*stream/depthwise_conv2d/depthwise:output:06stream/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.g
ReluRelu(stream/depthwise_conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:.�
 dense_1/Tensordot/ReadVariableOpReadVariableOp6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel*
_output_shapes

:(*
dtype0p
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".      �
dense_1/Tensordot/ReshapeReshapeRelu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:.(p
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .      (   �
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*&
_output_shapes
:.(�
dense_1/BiasAdd/ReadVariableOpReadVariableOp2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias*
_output_shapes
:(*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.(p
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*"
_output_shapes
:.(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:.(�
NoOpNoOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^stream/depthwise_conv2d/BiasAdd/ReadVariableOp1^stream/depthwise_conv2d/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:1(: : : : : 2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2`
.stream/depthwise_conv2d/BiasAdd/ReadVariableOp.stream/depthwise_conv2d/BiasAdd/ReadVariableOp2d
0stream/depthwise_conv2d/depthwise/ReadVariableOp0stream/depthwise_conv2d/depthwise/ReadVariableOp:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
�
�
@__inference_svdf_5_layer_call_and_return_conditional_losses_2697

inputsK
8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel:	@�p
Ustream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel:
�V
Gstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias:	�
identity��!dense_10/Tensordot/ReadVariableOp�2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp�4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:
@�
!dense_10/Tensordot/ReadVariableOpReadVariableOp8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel*
_output_shapes
:	@�*
dtype0q
 dense_10/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_10/Tensordot/ReshapeReshapeExpandDims:output:0)dense_10/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	
�q
dense_10/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      �   �
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0!dense_10/Tensordot/shape:output:0*
T0*'
_output_shapes
:
��
stream_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_5/PadPaddense_10/Tensordot:output:0stream_5/Pad/paddings:output:0*
T0*'
_output_shapes
:
��
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOpUstream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel*'
_output_shapes
:
�*
dtype0�
+stream_5/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      �      �
3stream_5/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_5/depthwise_conv2d_5/depthwiseDepthwiseConv2dNativestream_5/Pad:output:0<stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpReadVariableOpGstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias*
_output_shapes	
:�*
dtype0�
#stream_5/depthwise_conv2d_5/BiasAddBiasAdd.stream_5/depthwise_conv2d_5/depthwise:output:0:stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�l
ReluRelu,stream_5/depthwise_conv2d_5/BiasAdd:output:0*
T0*'
_output_shapes
:�k
SqueezeSqueezeRelu:activations:0*
T0*#
_output_shapes
:�*
squeeze_dims
[
IdentityIdentitySqueeze:output:0^NoOp*
T0*#
_output_shapes
:��
NoOpNoOp"^dense_10/Tensordot/ReadVariableOp3^stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp5^stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:
@: : : 2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2h
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2l
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:J F
"
_output_shapes
:
@
 
_user_specified_nameinputs
�	
�
%__inference_svdf_3_layer_call_fn_4577

inputs'
svdf_3_dense_6_kernel:@ :
 svdf_3_stream_3_depthwise_kernel:
 "
svdf_3_stream_3_bias: '
svdf_3_dense_7_kernel: @!
svdf_3_dense_7_bias:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_3_dense_6_kernel svdf_3_stream_3_depthwise_kernelsvdf_3_stream_3_biassvdf_3_dense_7_kernelsvdf_3_dense_7_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_3_layer_call_and_return_conditional_losses_2463j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�
�
'__inference_dense_11_layer_call_fn_4851

inputs"
dense_11_kernel:	�
dense_11_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsdense_11_kerneldense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_2567f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*"
_input_shapes
:	�: : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�
e
I__inference_speech_features_layer_call_and_return_conditional_losses_4297

inputs
identity_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       a
	transpose	Transposeinputstranspose/perm:output:0*
T0*
_output_shapes
:	�}�
AudioSpectrogramAudioSpectrogramtranspose:y:0*#
_output_shapes
:1�*
magnitude_squared(*
stride�*
window_size�S
Mfcc/sample_rateConst*
_output_shapes
: *
dtype0*
value
B :�}�
MfccMfccAudioSpectrogram:spectrogram:0Mfcc/sample_rate:output:0*"
_output_shapes
:1(*
dct_coefficient_count(*
filterbank_channel_countP*
upper_frequency_limit% ��E�
normalizer/sub/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�                                                                                                                                                                l
normalizer/subSubMfcc:output:0normalizer/sub/y:output:0*
T0*"
_output_shapes
:1(�
normalizer/truediv/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?}
normalizer/truedivRealDivnormalizer/sub:z:0normalizer/truediv/y:output:0*
T0*"
_output_shapes
:1(Y
IdentityIdentitynormalizer/truediv:z:0*
T0*"
_output_shapes
:1("
identityIdentity:output:0*
_input_shapes
:	�}:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�*
�
@__inference_svdf_3_layer_call_and_return_conditional_losses_2927

inputsH
6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel:@ o
Ustream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel:
 U
Gstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias: H
6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel: @@
2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias:@
identity�� dense_6/Tensordot/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp� dense_7/Tensordot/ReadVariableOp�2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp�4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_6/Tensordot/ReadVariableOpReadVariableOp6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel*
_output_shapes

:@ *
dtype0p
dense_6/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_6/Tensordot/ReshapeReshapeExpandDims:output:0(dense_6/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

: p
dense_6/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0 dense_6/Tensordot/shape:output:0*
T0*&
_output_shapes
: �
stream_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_3/PadPaddense_6/Tensordot:output:0stream_3/Pad/paddings:output:0*
T0*&
_output_shapes
: �
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpUstream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_3/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_3/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_3/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativestream_3/Pad:output:0<stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpGstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias*
_output_shapes
: *
dtype0�
#stream_3/depthwise_conv2d_3/BiasAddBiasAdd.stream_3/depthwise_conv2d_3/depthwise:output:0:stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_3/depthwise_conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_7/Tensordot/ReadVariableOpReadVariableOp6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel*
_output_shapes

: @*
dtype0p
dense_7/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_7/Tensordot/ReshapeReshapeRelu:activations:0(dense_7/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_7/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0 dense_7/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_7/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp3^stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp5^stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:@: : : : : 2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2h
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2l
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�	
�
B__inference_dense_11_layer_call_and_return_conditional_losses_2567

inputs8
%matmul_readvariableop_dense_11_kernel:	�2
$biasadd_readvariableop_dense_11_bias:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp|
MatMul/ReadVariableOpReadVariableOp%matmul_readvariableop_dense_11_kernel*
_output_shapes
:	�*
dtype0`
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:w
BiasAdd/ReadVariableOpReadVariableOp$biasadd_readvariableop_dense_11_bias*
_output_shapes
:*
dtype0m
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:	�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�
e
I__inference_speech_features_layer_call_and_return_conditional_losses_2300

inputs
identity_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       a
	transpose	Transposeinputstranspose/perm:output:0*
T0*
_output_shapes
:	�}�
AudioSpectrogramAudioSpectrogramtranspose:y:0*#
_output_shapes
:1�*
magnitude_squared(*
stride�*
window_size�S
Mfcc/sample_rateConst*
_output_shapes
: *
dtype0*
value
B :�}�
MfccMfccAudioSpectrogram:spectrogram:0Mfcc/sample_rate:output:0*"
_output_shapes
:1(*
dct_coefficient_count(*
filterbank_channel_countP*
upper_frequency_limit% ��E�
normalizer/sub/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�                                                                                                                                                                l
normalizer/subSubMfcc:output:0normalizer/sub/y:output:0*
T0*"
_output_shapes
:1(�
normalizer/truediv/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?}
normalizer/truedivRealDivnormalizer/sub:z:0normalizer/truediv/y:output:0*
T0*"
_output_shapes
:1(Y
IdentityIdentitynormalizer/truediv:z:0*
T0*"
_output_shapes
:1("
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:	�}:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�*
�
@__inference_svdf_3_layer_call_and_return_conditional_losses_4657

inputsH
6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel:@ o
Ustream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel:
 U
Gstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias: H
6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel: @@
2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias:@
identity�� dense_6/Tensordot/ReadVariableOp�dense_7/BiasAdd/ReadVariableOp� dense_7/Tensordot/ReadVariableOp�2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp�4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:@�
 dense_6/Tensordot/ReadVariableOpReadVariableOp6dense_6_tensordot_readvariableop_svdf_3_dense_6_kernel*
_output_shapes

:@ *
dtype0p
dense_6/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   �
dense_6/Tensordot/ReshapeReshapeExpandDims:output:0(dense_6/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:@�
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

: p
dense_6/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0 dense_6/Tensordot/shape:output:0*
T0*&
_output_shapes
: �
stream_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_3/PadPaddense_6/Tensordot:output:0stream_3/Pad/paddings:output:0*
T0*&
_output_shapes
: �
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpUstream_3_depthwise_conv2d_3_depthwise_readvariableop_svdf_3_stream_3_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_3/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_3/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_3/depthwise_conv2d_3/depthwiseDepthwiseConv2dNativestream_3/Pad:output:0<stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOpReadVariableOpGstream_3_depthwise_conv2d_3_biasadd_readvariableop_svdf_3_stream_3_bias*
_output_shapes
: *
dtype0�
#stream_3/depthwise_conv2d_3/BiasAddBiasAdd.stream_3/depthwise_conv2d_3/depthwise:output:0:stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_3/depthwise_conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_7/Tensordot/ReadVariableOpReadVariableOp6dense_7_tensordot_readvariableop_svdf_3_dense_7_kernel*
_output_shapes

: @*
dtype0p
dense_7/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_7/Tensordot/ReshapeReshapeRelu:activations:0(dense_7/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_7/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0 dense_7/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_7/BiasAdd/ReadVariableOpReadVariableOp2dense_7_biasadd_readvariableop_svdf_3_dense_7_bias*
_output_shapes
:@*
dtype0�
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_7/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_6/Tensordot/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp3^stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp5^stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:@: : : : : 2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2h
2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2stream_3/depthwise_conv2d_3/BiasAdd/ReadVariableOp2l
4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp4stream_3/depthwise_conv2d_3/depthwise/ReadVariableOp:J F
"
_output_shapes
:@
 
_user_specified_nameinputs
�
�

$__inference_model_layer_call_fn_2605
input_1%
svdf_0_dense_kernel:(8
svdf_0_stream_depthwise_kernel: 
svdf_0_stream_bias:'
svdf_0_dense_1_kernel:(!
svdf_0_dense_1_bias:('
svdf_1_dense_2_kernel:( :
 svdf_1_stream_1_depthwise_kernel:
 "
svdf_1_stream_1_bias: '
svdf_1_dense_3_kernel: (!
svdf_1_dense_3_bias:('
svdf_2_dense_4_kernel:( :
 svdf_2_stream_2_depthwise_kernel:
 "
svdf_2_stream_2_bias: '
svdf_2_dense_5_kernel: @!
svdf_2_dense_5_bias:@'
svdf_3_dense_6_kernel:@ :
 svdf_3_stream_3_depthwise_kernel:
 "
svdf_3_stream_3_bias: '
svdf_3_dense_7_kernel: @!
svdf_3_dense_7_bias:@'
svdf_4_dense_8_kernel:@@:
 svdf_4_stream_4_depthwise_kernel:
@"
svdf_4_stream_4_bias:@'
svdf_4_dense_9_kernel:@@!
svdf_4_dense_9_bias:@)
svdf_5_dense_10_kernel:	@�;
 svdf_5_stream_5_depthwise_kernel:
�#
svdf_5_stream_5_bias:	�"
dense_11_kernel:	�
dense_11_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1svdf_0_dense_kernelsvdf_0_stream_depthwise_kernelsvdf_0_stream_biassvdf_0_dense_1_kernelsvdf_0_dense_1_biassvdf_1_dense_2_kernel svdf_1_stream_1_depthwise_kernelsvdf_1_stream_1_biassvdf_1_dense_3_kernelsvdf_1_dense_3_biassvdf_2_dense_4_kernel svdf_2_stream_2_depthwise_kernelsvdf_2_stream_2_biassvdf_2_dense_5_kernelsvdf_2_dense_5_biassvdf_3_dense_6_kernel svdf_3_stream_3_depthwise_kernelsvdf_3_stream_3_biassvdf_3_dense_7_kernelsvdf_3_dense_7_biassvdf_4_dense_8_kernel svdf_4_stream_4_depthwise_kernelsvdf_4_stream_4_biassvdf_4_dense_9_kernelsvdf_4_dense_9_biassvdf_5_dense_10_kernel svdf_5_stream_5_depthwise_kernelsvdf_5_stream_5_biasdense_11_kerneldense_11_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2572f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	�}
!
_user_specified_name	input_1
�	
�
%__inference_svdf_2_layer_call_fn_4487

inputs'
svdf_2_dense_4_kernel:( :
 svdf_2_stream_2_depthwise_kernel:
 "
svdf_2_stream_2_bias: '
svdf_2_dense_5_kernel: @!
svdf_2_dense_5_bias:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_2_dense_4_kernel svdf_2_stream_2_depthwise_kernelsvdf_2_stream_2_biassvdf_2_dense_5_kernelsvdf_2_dense_5_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_2_layer_call_and_return_conditional_losses_2421j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:%(: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:%(
 
_user_specified_nameinputs
�*
�
@__inference_svdf_1_layer_call_and_return_conditional_losses_4442

inputsH
6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel:( o
Ustream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel:
 U
Gstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias: H
6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel: (@
2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias:(
identity�� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp�4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:.(�
 dense_2/Tensordot/ReadVariableOpReadVariableOp6dense_2_tensordot_readvariableop_svdf_1_dense_2_kernel*
_output_shapes

:( *
dtype0p
dense_2/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".   (   �
dense_2/Tensordot/ReshapeReshapeExpandDims:output:0(dense_2/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.(�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:. p
dense_2/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .          �
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0 dense_2/Tensordot/shape:output:0*
T0*&
_output_shapes
:. �
stream_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_1/PadPaddense_2/Tensordot:output:0stream_1/Pad/paddings:output:0*
T0*&
_output_shapes
:. �
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpUstream_1_depthwise_conv2d_1_depthwise_readvariableop_svdf_1_stream_1_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNativestream_1/Pad:output:0<stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:% *
paddingVALID*
strides
�
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGstream_1_depthwise_conv2d_1_biasadd_readvariableop_svdf_1_stream_1_bias*
_output_shapes
: *
dtype0�
#stream_1/depthwise_conv2d_1/BiasAddBiasAdd.stream_1/depthwise_conv2d_1/depthwise:output:0:stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:% k
ReluRelu,stream_1/depthwise_conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:% �
 dense_3/Tensordot/ReadVariableOpReadVariableOp6dense_3_tensordot_readvariableop_svdf_1_dense_3_kernel*
_output_shapes

: (*
dtype0p
dense_3/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%       �
dense_3/Tensordot/ReshapeReshapeRelu:activations:0(dense_3/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:% �
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:%(p
dense_3/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %      (   �
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0 dense_3/Tensordot/shape:output:0*
T0*&
_output_shapes
:%(�
dense_3/BiasAdd/ReadVariableOpReadVariableOp2dense_3_biasadd_readvariableop_svdf_1_dense_3_bias*
_output_shapes
:(*
dtype0�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:%(p
SqueezeSqueezedense_3/BiasAdd:output:0*
T0*"
_output_shapes
:%(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:%(�
NoOpNoOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp3^stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp5^stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:.(: : : : : 2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2h
2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2stream_1/depthwise_conv2d_1/BiasAdd/ReadVariableOp2l
4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp4stream_1/depthwise_conv2d_1/depthwise/ReadVariableOp:J F
"
_output_shapes
:.(
 
_user_specified_nameinputs
�
�
@__inference_svdf_5_layer_call_and_return_conditional_losses_2537

inputsK
8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel:	@�p
Ustream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel:
�V
Gstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias:	�
identity��!dense_10/Tensordot/ReadVariableOp�2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp�4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:
@�
!dense_10/Tensordot/ReadVariableOpReadVariableOp8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel*
_output_shapes
:	@�*
dtype0q
 dense_10/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_10/Tensordot/ReshapeReshapeExpandDims:output:0)dense_10/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	
�q
dense_10/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      �   �
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0!dense_10/Tensordot/shape:output:0*
T0*'
_output_shapes
:
��
stream_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_5/PadPaddense_10/Tensordot:output:0stream_5/Pad/paddings:output:0*
T0*'
_output_shapes
:
��
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOpUstream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel*'
_output_shapes
:
�*
dtype0�
+stream_5/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      �      �
3stream_5/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_5/depthwise_conv2d_5/depthwiseDepthwiseConv2dNativestream_5/Pad:output:0<stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpReadVariableOpGstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias*
_output_shapes	
:�*
dtype0�
#stream_5/depthwise_conv2d_5/BiasAddBiasAdd.stream_5/depthwise_conv2d_5/depthwise:output:0:stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�l
ReluRelu,stream_5/depthwise_conv2d_5/BiasAdd:output:0*
T0*'
_output_shapes
:�k
SqueezeSqueezeRelu:activations:0*
T0*#
_output_shapes
:�*
squeeze_dims
[
IdentityIdentitySqueeze:output:0^NoOp*
T0*#
_output_shapes
:��
NoOpNoOp"^dense_10/Tensordot/ReadVariableOp3^stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp5^stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:
@: : : 2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2h
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2l
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:J F
"
_output_shapes
:
@
 
_user_specified_nameinputs
�
�
@__inference_svdf_5_layer_call_and_return_conditional_losses_4788

inputsK
8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel:	@�p
Ustream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel:
�V
Gstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias:	�
identity��!dense_10/Tensordot/ReadVariableOp�2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp�4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:
@�
!dense_10/Tensordot/ReadVariableOpReadVariableOp8dense_10_tensordot_readvariableop_svdf_5_dense_10_kernel*
_output_shapes
:	@�*
dtype0q
 dense_10/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"
   @   �
dense_10/Tensordot/ReshapeReshapeExpandDims:output:0)dense_10/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:
@�
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes
:	
�q
dense_10/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   
      �   �
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0!dense_10/Tensordot/shape:output:0*
T0*'
_output_shapes
:
��
stream_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_5/PadPaddense_10/Tensordot:output:0stream_5/Pad/paddings:output:0*
T0*'
_output_shapes
:
��
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOpReadVariableOpUstream_5_depthwise_conv2d_5_depthwise_readvariableop_svdf_5_stream_5_depthwise_kernel*'
_output_shapes
:
�*
dtype0�
+stream_5/depthwise_conv2d_5/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
      �      �
3stream_5/depthwise_conv2d_5/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_5/depthwise_conv2d_5/depthwiseDepthwiseConv2dNativestream_5/Pad:output:0<stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:�*
paddingVALID*
strides
�
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOpReadVariableOpGstream_5_depthwise_conv2d_5_biasadd_readvariableop_svdf_5_stream_5_bias*
_output_shapes	
:�*
dtype0�
#stream_5/depthwise_conv2d_5/BiasAddBiasAdd.stream_5/depthwise_conv2d_5/depthwise:output:0:stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:�l
ReluRelu,stream_5/depthwise_conv2d_5/BiasAdd:output:0*
T0*'
_output_shapes
:�k
SqueezeSqueezeRelu:activations:0*
T0*#
_output_shapes
:�*
squeeze_dims
[
IdentityIdentitySqueeze:output:0^NoOp*
T0*#
_output_shapes
:��
NoOpNoOp"^dense_10/Tensordot/ReadVariableOp3^stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp5^stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*'
_input_shapes
:
@: : : 2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2h
2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2stream_5/depthwise_conv2d_5/BiasAdd/ReadVariableOp2l
4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp4stream_5/depthwise_conv2d_5/depthwise/ReadVariableOp:J F
"
_output_shapes
:
@
 
_user_specified_nameinputs
�
J
.__inference_speech_features_layer_call_fn_4266

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_speech_features_layer_call_and_return_conditional_losses_2300[
IdentityIdentityPartitionedCall:output:0*
T0*"
_output_shapes
:1("
identityIdentity:output:0*
_input_shapes
:	�}:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�*
�
@__inference_svdf_2_layer_call_and_return_conditional_losses_4532

inputsH
6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel:( o
Ustream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel:
 U
Gstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias: H
6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel: @@
2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias:@
identity�� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp�4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:%(�
 dense_4/Tensordot/ReadVariableOpReadVariableOp6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel*
_output_shapes

:( *
dtype0p
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%   (   �
dense_4/Tensordot/ReshapeReshapeExpandDims:output:0(dense_4/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:%(�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:% p
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %          �
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*&
_output_shapes
:% �
stream_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_2/PadPaddense_4/Tensordot:output:0stream_2/Pad/paddings:output:0*
T0*&
_output_shapes
:% �
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpUstream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_2/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_2/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_2/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativestream_2/Pad:output:0<stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias*
_output_shapes
: *
dtype0�
#stream_2/depthwise_conv2d_2/BiasAddBiasAdd.stream_2/depthwise_conv2d_2/depthwise:output:0:stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_2/depthwise_conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_5/Tensordot/ReadVariableOpReadVariableOp6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel*
_output_shapes

: @*
dtype0p
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_5/Tensordot/ReshapeReshapeRelu:activations:0(dense_5/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_5/BiasAdd/ReadVariableOpReadVariableOp2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias*
_output_shapes
:@*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_5/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp3^stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp5^stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:%(: : : : : 2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2h
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2l
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:J F
"
_output_shapes
:%(
 
_user_specified_nameinputs
�*
�
@__inference_svdf_2_layer_call_and_return_conditional_losses_2421

inputsH
6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel:( o
Ustream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel:
 U
Gstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias: H
6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel: @@
2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias:@
identity�� dense_4/Tensordot/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp�4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:%(�
 dense_4/Tensordot/ReadVariableOpReadVariableOp6dense_4_tensordot_readvariableop_svdf_2_dense_4_kernel*
_output_shapes

:( *
dtype0p
dense_4/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"%   (   �
dense_4/Tensordot/ReshapeReshapeExpandDims:output:0(dense_4/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:%(�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:% p
dense_4/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   %          �
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0 dense_4/Tensordot/shape:output:0*
T0*&
_output_shapes
:% �
stream_2/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 �
stream_2/PadPaddense_4/Tensordot:output:0stream_2/Pad/paddings:output:0*
T0*&
_output_shapes
:% �
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpUstream_2_depthwise_conv2d_2_depthwise_readvariableop_svdf_2_stream_2_depthwise_kernel*&
_output_shapes
:
 *
dtype0�
+stream_2/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"
             �
3stream_2/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
%stream_2/depthwise_conv2d_2/depthwiseDepthwiseConv2dNativestream_2/Pad:output:0<stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
�
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOpReadVariableOpGstream_2_depthwise_conv2d_2_biasadd_readvariableop_svdf_2_stream_2_bias*
_output_shapes
: *
dtype0�
#stream_2/depthwise_conv2d_2/BiasAddBiasAdd.stream_2/depthwise_conv2d_2/depthwise:output:0:stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
: k
ReluRelu,stream_2/depthwise_conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
: �
 dense_5/Tensordot/ReadVariableOpReadVariableOp6dense_5_tensordot_readvariableop_svdf_2_dense_5_kernel*
_output_shapes

: @*
dtype0p
dense_5/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       �
dense_5/Tensordot/ReshapeReshapeRelu:activations:0(dense_5/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

: �
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:@p
dense_5/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0 dense_5/Tensordot/shape:output:0*
T0*&
_output_shapes
:@�
dense_5/BiasAdd/ReadVariableOpReadVariableOp2dense_5_biasadd_readvariableop_svdf_2_dense_5_bias*
_output_shapes
:@*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@p
SqueezeSqueezedense_5/BiasAdd:output:0*
T0*"
_output_shapes
:@*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:@�
NoOpNoOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp3^stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp5^stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:%(: : : : : 2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2h
2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2stream_2/depthwise_conv2d_2/BiasAdd/ReadVariableOp2l
4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp4stream_2/depthwise_conv2d_2/depthwise/ReadVariableOp:J F
"
_output_shapes
:%(
 
_user_specified_nameinputs
�
B
&__inference_dropout_layer_call_fn_4834

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2636X
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	�"
identityIdentity:output:0*
_input_shapes
:	�:G C

_output_shapes
:	�
 
_user_specified_nameinputs
�
e
I__inference_speech_features_layer_call_and_return_conditional_losses_4284

inputs
identity_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       a
	transpose	Transposeinputstranspose/perm:output:0*
T0*
_output_shapes
:	�}�
AudioSpectrogramAudioSpectrogramtranspose:y:0*#
_output_shapes
:1�*
magnitude_squared(*
stride�*
window_size�S
Mfcc/sample_rateConst*
_output_shapes
: *
dtype0*
value
B :�}�
MfccMfccAudioSpectrogram:spectrogram:0Mfcc/sample_rate:output:0*"
_output_shapes
:1(*
dct_coefficient_count(*
filterbank_channel_countP*
upper_frequency_limit% ��E�
normalizer/sub/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�                                                                                                                                                                l
normalizer/subSubMfcc:output:0normalizer/sub/y:output:0*
T0*"
_output_shapes
:1(�
normalizer/truediv/yConst*
_output_shapes
:(*
dtype0*�
value�B�("�  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?  �?}
normalizer/truedivRealDivnormalizer/sub:z:0normalizer/truediv/y:output:0*
T0*"
_output_shapes
:1(Y
IdentityIdentitynormalizer/truediv:z:0*
T0*"
_output_shapes
:1("
identityIdentity:output:0*
_input_shapes
:	�}:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�)
�
@__inference_svdf_0_layer_call_and_return_conditional_losses_4352

inputsD
2dense_tensordot_readvariableop_svdf_0_dense_kernel:(i
Ostream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel:O
Astream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias:H
6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel:(@
2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias:(
identity��dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�.stream/depthwise_conv2d/BiasAdd/ReadVariableOp�0stream/depthwise_conv2d/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:1(�
dense/Tensordot/ReadVariableOpReadVariableOp2dense_tensordot_readvariableop_svdf_0_dense_kernel*
_output_shapes

:(*
dtype0n
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"1   (   �
dense/Tensordot/ReshapeReshapeExpandDims:output:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:1(�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:1n
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   1         �
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*&
_output_shapes
:1�
stream/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 z

stream/PadPaddense/Tensordot:output:0stream/Pad/paddings:output:0*
T0*&
_output_shapes
:1�
0stream/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpOstream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel*&
_output_shapes
:*
dtype0�
'stream/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
/stream/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
!stream/depthwise_conv2d/depthwiseDepthwiseConv2dNativestream/Pad:output:08stream/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:.*
paddingVALID*
strides
�
.stream/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpAstream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias*
_output_shapes
:*
dtype0�
stream/depthwise_conv2d/BiasAddBiasAdd*stream/depthwise_conv2d/depthwise:output:06stream/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.g
ReluRelu(stream/depthwise_conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:.�
 dense_1/Tensordot/ReadVariableOpReadVariableOp6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel*
_output_shapes

:(*
dtype0p
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".      �
dense_1/Tensordot/ReshapeReshapeRelu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:.(p
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .      (   �
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*&
_output_shapes
:.(�
dense_1/BiasAdd/ReadVariableOpReadVariableOp2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias*
_output_shapes
:(*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.(p
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*"
_output_shapes
:.(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:.(�
NoOpNoOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^stream/depthwise_conv2d/BiasAdd/ReadVariableOp1^stream/depthwise_conv2d/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:1(: : : : : 2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2`
.stream/depthwise_conv2d/BiasAdd/ReadVariableOp.stream/depthwise_conv2d/BiasAdd/ReadVariableOp2d
0stream/depthwise_conv2d/depthwise/ReadVariableOp0stream/depthwise_conv2d/depthwise/ReadVariableOp:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
�
�

$__inference_model_layer_call_fn_3670
input_1%
svdf_0_dense_kernel:(8
svdf_0_stream_depthwise_kernel: 
svdf_0_stream_bias:'
svdf_0_dense_1_kernel:(!
svdf_0_dense_1_bias:('
svdf_1_dense_2_kernel:( :
 svdf_1_stream_1_depthwise_kernel:
 "
svdf_1_stream_1_bias: '
svdf_1_dense_3_kernel: (!
svdf_1_dense_3_bias:('
svdf_2_dense_4_kernel:( :
 svdf_2_stream_2_depthwise_kernel:
 "
svdf_2_stream_2_bias: '
svdf_2_dense_5_kernel: @!
svdf_2_dense_5_bias:@'
svdf_3_dense_6_kernel:@ :
 svdf_3_stream_3_depthwise_kernel:
 "
svdf_3_stream_3_bias: '
svdf_3_dense_7_kernel: @!
svdf_3_dense_7_bias:@'
svdf_4_dense_8_kernel:@@:
 svdf_4_stream_4_depthwise_kernel:
@"
svdf_4_stream_4_bias:@'
svdf_4_dense_9_kernel:@@!
svdf_4_dense_9_bias:@)
svdf_5_dense_10_kernel:	@�;
 svdf_5_stream_5_depthwise_kernel:
�#
svdf_5_stream_5_bias:	�"
dense_11_kernel:	�
dense_11_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1svdf_0_dense_kernelsvdf_0_stream_depthwise_kernelsvdf_0_stream_biassvdf_0_dense_1_kernelsvdf_0_dense_1_biassvdf_1_dense_2_kernel svdf_1_stream_1_depthwise_kernelsvdf_1_stream_1_biassvdf_1_dense_3_kernelsvdf_1_dense_3_biassvdf_2_dense_4_kernel svdf_2_stream_2_depthwise_kernelsvdf_2_stream_2_biassvdf_2_dense_5_kernelsvdf_2_dense_5_biassvdf_3_dense_6_kernel svdf_3_stream_3_depthwise_kernelsvdf_3_stream_3_biassvdf_3_dense_7_kernelsvdf_3_dense_7_biassvdf_4_dense_8_kernel svdf_4_stream_4_depthwise_kernelsvdf_4_stream_4_biassvdf_4_dense_9_kernelsvdf_4_dense_9_biassvdf_5_dense_10_kernel svdf_5_stream_5_depthwise_kernelsvdf_5_stream_5_biasdense_11_kerneldense_11_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_3514f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes
:	�}
!
_user_specified_name	input_1
�)
�
@__inference_svdf_0_layer_call_and_return_conditional_losses_3305

inputsD
2dense_tensordot_readvariableop_svdf_0_dense_kernel:(i
Ostream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel:O
Astream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias:H
6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel:(@
2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias:(
identity��dense/Tensordot/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�.stream/depthwise_conv2d/BiasAdd/ReadVariableOp�0stream/depthwise_conv2d/depthwise/ReadVariableOpP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :j

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*&
_output_shapes
:1(�
dense/Tensordot/ReadVariableOpReadVariableOp2dense_tensordot_readvariableop_svdf_0_dense_kernel*
_output_shapes

:(*
dtype0n
dense/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"1   (   �
dense/Tensordot/ReshapeReshapeExpandDims:output:0&dense/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:1(�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:1n
dense/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   1         �
dense/TensordotReshape dense/Tensordot/MatMul:product:0dense/Tensordot/shape:output:0*
T0*&
_output_shapes
:1�
stream/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                                 z

stream/PadPaddense/Tensordot:output:0stream/Pad/paddings:output:0*
T0*&
_output_shapes
:1�
0stream/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpOstream_depthwise_conv2d_depthwise_readvariableop_svdf_0_stream_depthwise_kernel*&
_output_shapes
:*
dtype0�
'stream/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
/stream/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      �
!stream/depthwise_conv2d/depthwiseDepthwiseConv2dNativestream/Pad:output:08stream/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:.*
paddingVALID*
strides
�
.stream/depthwise_conv2d/BiasAdd/ReadVariableOpReadVariableOpAstream_depthwise_conv2d_biasadd_readvariableop_svdf_0_stream_bias*
_output_shapes
:*
dtype0�
stream/depthwise_conv2d/BiasAddBiasAdd*stream/depthwise_conv2d/depthwise:output:06stream/depthwise_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.g
ReluRelu(stream/depthwise_conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:.�
 dense_1/Tensordot/ReadVariableOpReadVariableOp6dense_1_tensordot_readvariableop_svdf_0_dense_1_kernel*
_output_shapes

:(*
dtype0p
dense_1/Tensordot/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB".      �
dense_1/Tensordot/ReshapeReshapeRelu:activations:0(dense_1/Tensordot/Reshape/shape:output:0*
T0*
_output_shapes

:.�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*
_output_shapes

:.(p
dense_1/Tensordot/shapeConst*
_output_shapes
:*
dtype0*%
valueB"   .      (   �
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0 dense_1/Tensordot/shape:output:0*
T0*&
_output_shapes
:.(�
dense_1/BiasAdd/ReadVariableOpReadVariableOp2dense_1_biasadd_readvariableop_svdf_0_dense_1_bias*
_output_shapes
:(*
dtype0�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:.(p
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*"
_output_shapes
:.(*
squeeze_dims
Z
IdentityIdentitySqueeze:output:0^NoOp*
T0*"
_output_shapes
:.(�
NoOpNoOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp/^stream/depthwise_conv2d/BiasAdd/ReadVariableOp1^stream/depthwise_conv2d/depthwise/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:1(: : : : : 2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2`
.stream/depthwise_conv2d/BiasAdd/ReadVariableOp.stream/depthwise_conv2d/BiasAdd/ReadVariableOp2d
0stream/depthwise_conv2d/depthwise/ReadVariableOp0stream/depthwise_conv2d/depthwise/ReadVariableOp:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
�
�

$__inference_model_layer_call_fn_3830

inputs%
svdf_0_dense_kernel:(8
svdf_0_stream_depthwise_kernel: 
svdf_0_stream_bias:'
svdf_0_dense_1_kernel:(!
svdf_0_dense_1_bias:('
svdf_1_dense_2_kernel:( :
 svdf_1_stream_1_depthwise_kernel:
 "
svdf_1_stream_1_bias: '
svdf_1_dense_3_kernel: (!
svdf_1_dense_3_bias:('
svdf_2_dense_4_kernel:( :
 svdf_2_stream_2_depthwise_kernel:
 "
svdf_2_stream_2_bias: '
svdf_2_dense_5_kernel: @!
svdf_2_dense_5_bias:@'
svdf_3_dense_6_kernel:@ :
 svdf_3_stream_3_depthwise_kernel:
 "
svdf_3_stream_3_bias: '
svdf_3_dense_7_kernel: @!
svdf_3_dense_7_bias:@'
svdf_4_dense_8_kernel:@@:
 svdf_4_stream_4_depthwise_kernel:
@"
svdf_4_stream_4_bias:@'
svdf_4_dense_9_kernel:@@!
svdf_4_dense_9_bias:@)
svdf_5_dense_10_kernel:	@�;
 svdf_5_stream_5_depthwise_kernel:
�#
svdf_5_stream_5_bias:	�"
dense_11_kernel:	�
dense_11_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_0_dense_kernelsvdf_0_stream_depthwise_kernelsvdf_0_stream_biassvdf_0_dense_1_kernelsvdf_0_dense_1_biassvdf_1_dense_2_kernel svdf_1_stream_1_depthwise_kernelsvdf_1_stream_1_biassvdf_1_dense_3_kernelsvdf_1_dense_3_biassvdf_2_dense_4_kernel svdf_2_stream_2_depthwise_kernelsvdf_2_stream_2_biassvdf_2_dense_5_kernelsvdf_2_dense_5_biassvdf_3_dense_6_kernel svdf_3_stream_3_depthwise_kernelsvdf_3_stream_3_biassvdf_3_dense_7_kernelsvdf_3_dense_7_biassvdf_4_dense_8_kernel svdf_4_stream_4_depthwise_kernelsvdf_4_stream_4_biassvdf_4_dense_9_kernelsvdf_4_dense_9_biassvdf_5_dense_10_kernel svdf_5_stream_5_depthwise_kernelsvdf_5_stream_5_biasdense_11_kerneldense_11_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_2572f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�
�

$__inference_model_layer_call_fn_3865

inputs%
svdf_0_dense_kernel:(8
svdf_0_stream_depthwise_kernel: 
svdf_0_stream_bias:'
svdf_0_dense_1_kernel:(!
svdf_0_dense_1_bias:('
svdf_1_dense_2_kernel:( :
 svdf_1_stream_1_depthwise_kernel:
 "
svdf_1_stream_1_bias: '
svdf_1_dense_3_kernel: (!
svdf_1_dense_3_bias:('
svdf_2_dense_4_kernel:( :
 svdf_2_stream_2_depthwise_kernel:
 "
svdf_2_stream_2_bias: '
svdf_2_dense_5_kernel: @!
svdf_2_dense_5_bias:@'
svdf_3_dense_6_kernel:@ :
 svdf_3_stream_3_depthwise_kernel:
 "
svdf_3_stream_3_bias: '
svdf_3_dense_7_kernel: @!
svdf_3_dense_7_bias:@'
svdf_4_dense_8_kernel:@@:
 svdf_4_stream_4_depthwise_kernel:
@"
svdf_4_stream_4_bias:@'
svdf_4_dense_9_kernel:@@!
svdf_4_dense_9_bias:@)
svdf_5_dense_10_kernel:	@�;
 svdf_5_stream_5_depthwise_kernel:
�#
svdf_5_stream_5_bias:	�"
dense_11_kernel:	�
dense_11_bias:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_0_dense_kernelsvdf_0_stream_depthwise_kernelsvdf_0_stream_biassvdf_0_dense_1_kernelsvdf_0_dense_1_biassvdf_1_dense_2_kernel svdf_1_stream_1_depthwise_kernelsvdf_1_stream_1_biassvdf_1_dense_3_kernelsvdf_1_dense_3_biassvdf_2_dense_4_kernel svdf_2_stream_2_depthwise_kernelsvdf_2_stream_2_biassvdf_2_dense_5_kernelsvdf_2_dense_5_biassvdf_3_dense_6_kernel svdf_3_stream_3_depthwise_kernelsvdf_3_stream_3_biassvdf_3_dense_7_kernelsvdf_3_dense_7_biassvdf_4_dense_8_kernel svdf_4_stream_4_depthwise_kernelsvdf_4_stream_4_biassvdf_4_dense_9_kernelsvdf_4_dense_9_biassvdf_5_dense_10_kernel svdf_5_stream_5_depthwise_kernelsvdf_5_stream_5_biasdense_11_kerneldense_11_bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_3514f
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	�}
 
_user_specified_nameinputs
�9
�
?__inference_model_layer_call_and_return_conditional_losses_3714
input_1,
svdf_0_svdf_0_dense_kernel:(?
%svdf_0_svdf_0_stream_depthwise_kernel:'
svdf_0_svdf_0_stream_bias:.
svdf_0_svdf_0_dense_1_kernel:((
svdf_0_svdf_0_dense_1_bias:(.
svdf_1_svdf_1_dense_2_kernel:( A
'svdf_1_svdf_1_stream_1_depthwise_kernel:
 )
svdf_1_svdf_1_stream_1_bias: .
svdf_1_svdf_1_dense_3_kernel: ((
svdf_1_svdf_1_dense_3_bias:(.
svdf_2_svdf_2_dense_4_kernel:( A
'svdf_2_svdf_2_stream_2_depthwise_kernel:
 )
svdf_2_svdf_2_stream_2_bias: .
svdf_2_svdf_2_dense_5_kernel: @(
svdf_2_svdf_2_dense_5_bias:@.
svdf_3_svdf_3_dense_6_kernel:@ A
'svdf_3_svdf_3_stream_3_depthwise_kernel:
 )
svdf_3_svdf_3_stream_3_bias: .
svdf_3_svdf_3_dense_7_kernel: @(
svdf_3_svdf_3_dense_7_bias:@.
svdf_4_svdf_4_dense_8_kernel:@@A
'svdf_4_svdf_4_stream_4_depthwise_kernel:
@)
svdf_4_svdf_4_stream_4_bias:@.
svdf_4_svdf_4_dense_9_kernel:@@(
svdf_4_svdf_4_dense_9_bias:@0
svdf_5_svdf_5_dense_10_kernel:	@�B
'svdf_5_svdf_5_stream_5_depthwise_kernel:
�*
svdf_5_svdf_5_stream_5_bias:	�+
dense_11_dense_11_kernel:	�$
dense_11_dense_11_bias:
identity�� dense_11/StatefulPartitionedCall�svdf_0/StatefulPartitionedCall�svdf_1/StatefulPartitionedCall�svdf_2/StatefulPartitionedCall�svdf_3/StatefulPartitionedCall�svdf_4/StatefulPartitionedCall�svdf_5/StatefulPartitionedCall�
speech_features/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_speech_features_layer_call_and_return_conditional_losses_2300�
svdf_0/StatefulPartitionedCallStatefulPartitionedCall(speech_features/PartitionedCall:output:0svdf_0_svdf_0_dense_kernel%svdf_0_svdf_0_stream_depthwise_kernelsvdf_0_svdf_0_stream_biassvdf_0_svdf_0_dense_1_kernelsvdf_0_svdf_0_dense_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:.(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_0_layer_call_and_return_conditional_losses_2337�
svdf_1/StatefulPartitionedCallStatefulPartitionedCall'svdf_0/StatefulPartitionedCall:output:0svdf_1_svdf_1_dense_2_kernel'svdf_1_svdf_1_stream_1_depthwise_kernelsvdf_1_svdf_1_stream_1_biassvdf_1_svdf_1_dense_3_kernelsvdf_1_svdf_1_dense_3_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:%(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_1_layer_call_and_return_conditional_losses_2379�
svdf_2/StatefulPartitionedCallStatefulPartitionedCall'svdf_1/StatefulPartitionedCall:output:0svdf_2_svdf_2_dense_4_kernel'svdf_2_svdf_2_stream_2_depthwise_kernelsvdf_2_svdf_2_stream_2_biassvdf_2_svdf_2_dense_5_kernelsvdf_2_svdf_2_dense_5_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_2_layer_call_and_return_conditional_losses_2421�
svdf_3/StatefulPartitionedCallStatefulPartitionedCall'svdf_2/StatefulPartitionedCall:output:0svdf_3_svdf_3_dense_6_kernel'svdf_3_svdf_3_stream_3_depthwise_kernelsvdf_3_svdf_3_stream_3_biassvdf_3_svdf_3_dense_7_kernelsvdf_3_svdf_3_dense_7_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_3_layer_call_and_return_conditional_losses_2463�
svdf_4/StatefulPartitionedCallStatefulPartitionedCall'svdf_3/StatefulPartitionedCall:output:0svdf_4_svdf_4_dense_8_kernel'svdf_4_svdf_4_stream_4_depthwise_kernelsvdf_4_svdf_4_stream_4_biassvdf_4_svdf_4_dense_9_kernelsvdf_4_svdf_4_dense_9_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:
@*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_4_layer_call_and_return_conditional_losses_2505�
svdf_5/StatefulPartitionedCallStatefulPartitionedCall'svdf_4/StatefulPartitionedCall:output:0svdf_5_svdf_5_dense_10_kernel'svdf_5_svdf_5_stream_5_depthwise_kernelsvdf_5_svdf_5_stream_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:�*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_5_layer_call_and_return_conditional_losses_2537�
stream_6/PartitionedCallPartitionedCall'svdf_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_stream_6_layer_call_and_return_conditional_losses_2548�
dropout/PartitionedCallPartitionedCall!stream_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_2555�
 dense_11/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_11_dense_11_kerneldense_11_dense_11_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_11_layer_call_and_return_conditional_losses_2567o
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:�
NoOpNoOp!^dense_11/StatefulPartitionedCall^svdf_0/StatefulPartitionedCall^svdf_1/StatefulPartitionedCall^svdf_2/StatefulPartitionedCall^svdf_3/StatefulPartitionedCall^svdf_4/StatefulPartitionedCall^svdf_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*Z
_input_shapesI
G:	�}: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2@
svdf_0/StatefulPartitionedCallsvdf_0/StatefulPartitionedCall2@
svdf_1/StatefulPartitionedCallsvdf_1/StatefulPartitionedCall2@
svdf_2/StatefulPartitionedCallsvdf_2/StatefulPartitionedCall2@
svdf_3/StatefulPartitionedCallsvdf_3/StatefulPartitionedCall2@
svdf_4/StatefulPartitionedCallsvdf_4/StatefulPartitionedCall2@
svdf_5/StatefulPartitionedCallsvdf_5/StatefulPartitionedCall:H D

_output_shapes
:	�}
!
_user_specified_name	input_1
�	
�
%__inference_svdf_0_layer_call_fn_4307

inputs%
svdf_0_dense_kernel:(8
svdf_0_stream_depthwise_kernel: 
svdf_0_stream_bias:'
svdf_0_dense_1_kernel:(!
svdf_0_dense_1_bias:(
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputssvdf_0_dense_kernelsvdf_0_stream_depthwise_kernelsvdf_0_stream_biassvdf_0_dense_1_kernelsvdf_0_dense_1_bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:.(*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_svdf_0_layer_call_and_return_conditional_losses_2337j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:.(`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:1(: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
�@
�
__inference__traced_save_4974
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop2
.savev2_svdf_0_dense_kernel_read_readvariableop=
9savev2_svdf_0_stream_depthwise_kernel_read_readvariableop1
-savev2_svdf_0_stream_bias_read_readvariableop4
0savev2_svdf_0_dense_1_kernel_read_readvariableop2
.savev2_svdf_0_dense_1_bias_read_readvariableop4
0savev2_svdf_1_dense_2_kernel_read_readvariableop?
;savev2_svdf_1_stream_1_depthwise_kernel_read_readvariableop3
/savev2_svdf_1_stream_1_bias_read_readvariableop4
0savev2_svdf_1_dense_3_kernel_read_readvariableop2
.savev2_svdf_1_dense_3_bias_read_readvariableop4
0savev2_svdf_2_dense_4_kernel_read_readvariableop?
;savev2_svdf_2_stream_2_depthwise_kernel_read_readvariableop3
/savev2_svdf_2_stream_2_bias_read_readvariableop4
0savev2_svdf_2_dense_5_kernel_read_readvariableop2
.savev2_svdf_2_dense_5_bias_read_readvariableop4
0savev2_svdf_3_dense_6_kernel_read_readvariableop?
;savev2_svdf_3_stream_3_depthwise_kernel_read_readvariableop3
/savev2_svdf_3_stream_3_bias_read_readvariableop4
0savev2_svdf_3_dense_7_kernel_read_readvariableop2
.savev2_svdf_3_dense_7_bias_read_readvariableop4
0savev2_svdf_4_dense_8_kernel_read_readvariableop?
;savev2_svdf_4_stream_4_depthwise_kernel_read_readvariableop3
/savev2_svdf_4_stream_4_bias_read_readvariableop4
0savev2_svdf_4_dense_9_kernel_read_readvariableop2
.savev2_svdf_4_dense_9_bias_read_readvariableop5
1savev2_svdf_5_dense_10_kernel_read_readvariableop?
;savev2_svdf_5_stream_5_depthwise_kernel_read_readvariableop3
/savev2_svdf_5_stream_5_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop.savev2_svdf_0_dense_kernel_read_readvariableop9savev2_svdf_0_stream_depthwise_kernel_read_readvariableop-savev2_svdf_0_stream_bias_read_readvariableop0savev2_svdf_0_dense_1_kernel_read_readvariableop.savev2_svdf_0_dense_1_bias_read_readvariableop0savev2_svdf_1_dense_2_kernel_read_readvariableop;savev2_svdf_1_stream_1_depthwise_kernel_read_readvariableop/savev2_svdf_1_stream_1_bias_read_readvariableop0savev2_svdf_1_dense_3_kernel_read_readvariableop.savev2_svdf_1_dense_3_bias_read_readvariableop0savev2_svdf_2_dense_4_kernel_read_readvariableop;savev2_svdf_2_stream_2_depthwise_kernel_read_readvariableop/savev2_svdf_2_stream_2_bias_read_readvariableop0savev2_svdf_2_dense_5_kernel_read_readvariableop.savev2_svdf_2_dense_5_bias_read_readvariableop0savev2_svdf_3_dense_6_kernel_read_readvariableop;savev2_svdf_3_stream_3_depthwise_kernel_read_readvariableop/savev2_svdf_3_stream_3_bias_read_readvariableop0savev2_svdf_3_dense_7_kernel_read_readvariableop.savev2_svdf_3_dense_7_bias_read_readvariableop0savev2_svdf_4_dense_8_kernel_read_readvariableop;savev2_svdf_4_stream_4_depthwise_kernel_read_readvariableop/savev2_svdf_4_stream_4_bias_read_readvariableop0savev2_svdf_4_dense_9_kernel_read_readvariableop.savev2_svdf_4_dense_9_bias_read_readvariableop1savev2_svdf_5_dense_10_kernel_read_readvariableop;savev2_svdf_5_stream_5_depthwise_kernel_read_readvariableop/savev2_svdf_5_stream_5_bias_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *-
dtypes#
!2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�::(:::(:(:( :
 : : (:(:( :
 : : @:@:@ :
 : : @:@:@@:
@:@:@@:@:	@�:
�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�: 

_output_shapes
::$ 

_output_shapes

:(:,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:(: 

_output_shapes
:(:$ 

_output_shapes

:( :,	(
&
_output_shapes
:
 : 


_output_shapes
: :$ 

_output_shapes

: (: 

_output_shapes
:(:$ 

_output_shapes

:( :,(
&
_output_shapes
:
 : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@ :,(
&
_output_shapes
:
 : 

_output_shapes
: :$ 

_output_shapes

: @: 

_output_shapes
:@:$ 

_output_shapes

:@@:,(
&
_output_shapes
:
@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	@�:-)
'
_output_shapes
:
�:!

_output_shapes	
:�:

_output_shapes
: "�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
3
input_1(
serving_default_input_1:0	�}3
dense_11'
StatefulPartitionedCall:0tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

params

rand_shift
rand_stretch_squeeze

data_frame
	add_noise
preemphasis
 	windowing
!mag_rdft_mel
"log_max
#dct
$
normalizer
%spec_augment
&spec_cutout"
_tf_keras_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-dropout1

.dense1
/
depth_cnn1

0dense2
1
batch_norm"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8dropout1

9dense1
:
depth_cnn1

;dense2
<
batch_norm"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Cdropout1

Ddense1
E
depth_cnn1

Fdense2
G
batch_norm"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
Ndropout1

Odense1
P
depth_cnn1

Qdense2
R
batch_norm"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Ydropout1

Zdense1
[
depth_cnn1

\dense2
]
batch_norm"
_tf_keras_layer
�
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses
ddropout1

edense1
f
depth_cnn1

gdense2
h
batch_norm"
_tf_keras_layer
�
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
ocell
pstate_shape"
_tf_keras_layer
�
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses
w_random_generator"
_tf_keras_layer
�
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses

~kernel
bias"
_tf_keras_layer
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
~28
29"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
~28
29"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
$__inference_model_layer_call_fn_2605
$__inference_model_layer_call_fn_3830
$__inference_model_layer_call_fn_3865
$__inference_model_layer_call_fn_3670�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
?__inference_model_layer_call_and_return_conditional_losses_4063
?__inference_model_layer_call_and_return_conditional_losses_4261
?__inference_model_layer_call_and_return_conditional_losses_3714
?__inference_model_layer_call_and_return_conditional_losses_3758�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
__inference__wrapped_model_2280input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_speech_features_layer_call_fn_4266
.__inference_speech_features_layer_call_fn_4271�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_speech_features_layer_call_and_return_conditional_losses_4284
I__inference_speech_features_layer_call_and_return_conditional_losses_4297�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z�trace_0z�trace_1
 "
trackable_dict_wrapper
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
=
�	keras_api
�padding_layer"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
)
�	keras_api"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�mean
�stddev"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
%__inference_svdf_0_layer_call_fn_4307
%__inference_svdf_0_layer_call_fn_4317�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
@__inference_svdf_0_layer_call_and_return_conditional_losses_4352
@__inference_svdf_0_layer_call_and_return_conditional_losses_4387�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
%__inference_svdf_1_layer_call_fn_4397
%__inference_svdf_1_layer_call_fn_4407�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
@__inference_svdf_1_layer_call_and_return_conditional_losses_4442
@__inference_svdf_1_layer_call_and_return_conditional_losses_4477�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
%__inference_svdf_2_layer_call_fn_4487
%__inference_svdf_2_layer_call_fn_4497�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
@__inference_svdf_2_layer_call_and_return_conditional_losses_4532
@__inference_svdf_2_layer_call_and_return_conditional_losses_4567�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
%__inference_svdf_3_layer_call_fn_4577
%__inference_svdf_3_layer_call_fn_4587�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
@__inference_svdf_3_layer_call_and_return_conditional_losses_4622
@__inference_svdf_3_layer_call_and_return_conditional_losses_4657�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
%__inference_svdf_4_layer_call_fn_4667
%__inference_svdf_4_layer_call_fn_4677�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
@__inference_svdf_4_layer_call_and_return_conditional_losses_4712
@__inference_svdf_4_layer_call_and_return_conditional_losses_4747�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
8
�0
�1
�2"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
%__inference_svdf_5_layer_call_fn_4755
%__inference_svdf_5_layer_call_fn_4763�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
@__inference_svdf_5_layer_call_and_return_conditional_losses_4788
@__inference_svdf_5_layer_call_and_return_conditional_losses_4813�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�cell
�state_shape"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_stream_6_layer_call_fn_4818�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_stream_6_layer_call_and_return_conditional_losses_4824�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_dropout_layer_call_fn_4829
&__inference_dropout_layer_call_fn_4834�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_dropout_layer_call_and_return_conditional_losses_4839
A__inference_dropout_layer_call_and_return_conditional_losses_4844�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_11_layer_call_fn_4851�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
B__inference_dense_11_layer_call_and_return_conditional_losses_4861�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�2dense_11/kernel
:2dense_11/bias
%:#(2svdf_0/dense/kernel
8:62svdf_0/stream/depthwise_kernel
 :2svdf_0/stream/bias
':%(2svdf_0/dense_1/kernel
!:(2svdf_0/dense_1/bias
':%( 2svdf_1/dense_2/kernel
::8
 2 svdf_1/stream_1/depthwise_kernel
":  2svdf_1/stream_1/bias
':% (2svdf_1/dense_3/kernel
!:(2svdf_1/dense_3/bias
':%( 2svdf_2/dense_4/kernel
::8
 2 svdf_2/stream_2/depthwise_kernel
":  2svdf_2/stream_2/bias
':% @2svdf_2/dense_5/kernel
!:@2svdf_2/dense_5/bias
':%@ 2svdf_3/dense_6/kernel
::8
 2 svdf_3/stream_3/depthwise_kernel
":  2svdf_3/stream_3/bias
':% @2svdf_3/dense_7/kernel
!:@2svdf_3/dense_7/bias
':%@@2svdf_4/dense_8/kernel
::8
@2 svdf_4/stream_4/depthwise_kernel
": @2svdf_4/stream_4/bias
':%@@2svdf_4/dense_9/kernel
!:@2svdf_4/dense_9/bias
):'	@�2svdf_5/dense_10/kernel
;:9
�2 svdf_5/stream_5/depthwise_kernel
#:!�2svdf_5/stream_5/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_model_layer_call_fn_2605input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3830inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3865inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_model_layer_call_fn_3670input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_4063inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_4261inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_3714input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_model_layer_call_and_return_conditional_losses_3758input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
"__inference_signature_wrapper_3795input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
v
0
1
2
3
4
 5
!6
"7
#8
$9
%10
&11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_speech_features_layer_call_fn_4266inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
.__inference_speech_features_layer_call_fn_4271inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
I__inference_speech_features_layer_call_and_return_conditional_losses_4284inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
I__inference_speech_features_layer_call_and_return_conditional_losses_4297inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
)
�	keras_api"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
C
-0
.1
/2
03
14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_svdf_0_layer_call_fn_4307inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_svdf_0_layer_call_fn_4317inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_0_layer_call_and_return_conditional_losses_4352inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_0_layer_call_and_return_conditional_losses_4387inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
C
80
91
:2
;3
<4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_svdf_1_layer_call_fn_4397inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_svdf_1_layer_call_fn_4407inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_1_layer_call_and_return_conditional_losses_4442inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_1_layer_call_and_return_conditional_losses_4477inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
C
C0
D1
E2
F3
G4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_svdf_2_layer_call_fn_4487inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_svdf_2_layer_call_fn_4497inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_2_layer_call_and_return_conditional_losses_4532inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_2_layer_call_and_return_conditional_losses_4567inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
C
N0
O1
P2
Q3
R4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_svdf_3_layer_call_fn_4577inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_svdf_3_layer_call_fn_4587inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_3_layer_call_and_return_conditional_losses_4622inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_3_layer_call_and_return_conditional_losses_4657inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
C
Y0
Z1
[2
\3
]4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_svdf_4_layer_call_fn_4667inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_svdf_4_layer_call_fn_4677inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_4_layer_call_and_return_conditional_losses_4712inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_4_layer_call_and_return_conditional_losses_4747inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
C
d0
e1
f2
g3
h4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_svdf_5_layer_call_fn_4755inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_svdf_5_layer_call_fn_4763inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_5_layer_call_and_return_conditional_losses_4788inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_svdf_5_layer_call_and_return_conditional_losses_4813inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�depthwise_kernel
	�bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_stream_6_layer_call_fn_4818inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_stream_6_layer_call_and_return_conditional_losses_4824inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_dropout_layer_call_fn_4829inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_dropout_layer_call_fn_4834inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dropout_layer_call_and_return_conditional_losses_4839inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_dropout_layer_call_and_return_conditional_losses_4844inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dense_11_layer_call_fn_4851inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dense_11_layer_call_and_return_conditional_losses_4861inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
__inference__wrapped_model_2280�:����������������������������~(�%
�
�
input_1	�}
� "*�'
%
dense_11�
dense_11�
B__inference_dense_11_layer_call_and_return_conditional_losses_4861R~'�$
�
�
inputs	�
� "#� 
�
tensor_0
� r
'__inference_dense_11_layer_call_fn_4851G~'�$
�
�
inputs	�
� "�
unknown�
A__inference_dropout_layer_call_and_return_conditional_losses_4839S+�(
!�
�
inputs	�
p 
� "$�!
�
tensor_0	�
� �
A__inference_dropout_layer_call_and_return_conditional_losses_4844S+�(
!�
�
inputs	�
p
� "$�!
�
tensor_0	�
� r
&__inference_dropout_layer_call_fn_4829H+�(
!�
�
inputs	�
p 
� "�
unknown	�r
&__inference_dropout_layer_call_fn_4834H+�(
!�
�
inputs	�
p
� "�
unknown	��
?__inference_model_layer_call_and_return_conditional_losses_3714�:����������������������������~0�-
&�#
�
input_1	�}
p 

 
� "#� 
�
tensor_0
� �
?__inference_model_layer_call_and_return_conditional_losses_3758�:����������������������������~0�-
&�#
�
input_1	�}
p

 
� "#� 
�
tensor_0
� �
?__inference_model_layer_call_and_return_conditional_losses_4063�:����������������������������~/�,
%�"
�
inputs	�}
p 

 
� "#� 
�
tensor_0
� �
?__inference_model_layer_call_and_return_conditional_losses_4261�:����������������������������~/�,
%�"
�
inputs	�}
p

 
� "#� 
�
tensor_0
� �
$__inference_model_layer_call_fn_2605�:����������������������������~0�-
&�#
�
input_1	�}
p 

 
� "�
unknown�
$__inference_model_layer_call_fn_3670�:����������������������������~0�-
&�#
�
input_1	�}
p

 
� "�
unknown�
$__inference_model_layer_call_fn_3830�:����������������������������~/�,
%�"
�
inputs	�}
p 

 
� "�
unknown�
$__inference_model_layer_call_fn_3865�:����������������������������~/�,
%�"
�
inputs	�}
p

 
� "�
unknown�
"__inference_signature_wrapper_3795�:����������������������������~3�0
� 
)�&
$
input_1�
input_1	�}"*�'
%
dense_11�
dense_11�
I__inference_speech_features_layer_call_and_return_conditional_losses_4284b7�4
�
�
inputs	�}
�

trainingp "'�$
�
tensor_01(
� �
I__inference_speech_features_layer_call_and_return_conditional_losses_4297b7�4
�
�
inputs	�}
�

trainingp"'�$
�
tensor_01(
� �
.__inference_speech_features_layer_call_fn_4266W7�4
�
�
inputs	�}
�

trainingp "�
unknown1(�
.__inference_speech_features_layer_call_fn_4271W7�4
�
�
inputs	�}
�

trainingp"�
unknown1(�
B__inference_stream_6_layer_call_and_return_conditional_losses_4824S+�(
!�
�
inputs�
� "$�!
�
tensor_0	�
� s
'__inference_stream_6_layer_call_fn_4818H+�(
!�
�
inputs�
� "�
unknown	��
@__inference_svdf_0_layer_call_and_return_conditional_losses_4352e
�����.�+
$�!
�
inputs1(
p 
� "'�$
�
tensor_0.(
� �
@__inference_svdf_0_layer_call_and_return_conditional_losses_4387e
�����.�+
$�!
�
inputs1(
p
� "'�$
�
tensor_0.(
� �
%__inference_svdf_0_layer_call_fn_4307Z
�����.�+
$�!
�
inputs1(
p 
� "�
unknown.(�
%__inference_svdf_0_layer_call_fn_4317Z
�����.�+
$�!
�
inputs1(
p
� "�
unknown.(�
@__inference_svdf_1_layer_call_and_return_conditional_losses_4442e
�����.�+
$�!
�
inputs.(
p 
� "'�$
�
tensor_0%(
� �
@__inference_svdf_1_layer_call_and_return_conditional_losses_4477e
�����.�+
$�!
�
inputs.(
p
� "'�$
�
tensor_0%(
� �
%__inference_svdf_1_layer_call_fn_4397Z
�����.�+
$�!
�
inputs.(
p 
� "�
unknown%(�
%__inference_svdf_1_layer_call_fn_4407Z
�����.�+
$�!
�
inputs.(
p
� "�
unknown%(�
@__inference_svdf_2_layer_call_and_return_conditional_losses_4532e
�����.�+
$�!
�
inputs%(
p 
� "'�$
�
tensor_0@
� �
@__inference_svdf_2_layer_call_and_return_conditional_losses_4567e
�����.�+
$�!
�
inputs%(
p
� "'�$
�
tensor_0@
� �
%__inference_svdf_2_layer_call_fn_4487Z
�����.�+
$�!
�
inputs%(
p 
� "�
unknown@�
%__inference_svdf_2_layer_call_fn_4497Z
�����.�+
$�!
�
inputs%(
p
� "�
unknown@�
@__inference_svdf_3_layer_call_and_return_conditional_losses_4622e
�����.�+
$�!
�
inputs@
p 
� "'�$
�
tensor_0@
� �
@__inference_svdf_3_layer_call_and_return_conditional_losses_4657e
�����.�+
$�!
�
inputs@
p
� "'�$
�
tensor_0@
� �
%__inference_svdf_3_layer_call_fn_4577Z
�����.�+
$�!
�
inputs@
p 
� "�
unknown@�
%__inference_svdf_3_layer_call_fn_4587Z
�����.�+
$�!
�
inputs@
p
� "�
unknown@�
@__inference_svdf_4_layer_call_and_return_conditional_losses_4712e
�����.�+
$�!
�
inputs@
p 
� "'�$
�
tensor_0
@
� �
@__inference_svdf_4_layer_call_and_return_conditional_losses_4747e
�����.�+
$�!
�
inputs@
p
� "'�$
�
tensor_0
@
� �
%__inference_svdf_4_layer_call_fn_4667Z
�����.�+
$�!
�
inputs@
p 
� "�
unknown
@�
%__inference_svdf_4_layer_call_fn_4677Z
�����.�+
$�!
�
inputs@
p
� "�
unknown
@�
@__inference_svdf_5_layer_call_and_return_conditional_losses_4788b���.�+
$�!
�
inputs
@
p 
� "(�%
�
tensor_0�
� �
@__inference_svdf_5_layer_call_and_return_conditional_losses_4813b���.�+
$�!
�
inputs
@
p
� "(�%
�
tensor_0�
� �
%__inference_svdf_5_layer_call_fn_4755W���.�+
$�!
�
inputs
@
p 
� "�
unknown��
%__inference_svdf_5_layer_call_fn_4763W���.�+
$�!
�
inputs
@
p
� "�
unknown�