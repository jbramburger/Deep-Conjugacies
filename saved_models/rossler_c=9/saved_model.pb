??%
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
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
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??!
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
h

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

:d*
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
_output_shapes
:d*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:d*
dtype0
r
dense_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_35/bias
k
!dense_35/bias/Read/ReadVariableOpReadVariableOpdense_35/bias*
_output_shapes
:*
dtype0
z
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_36/kernel
s
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes

:d*
dtype0
r
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_namedense_36/bias
k
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes
:d*
dtype0
z
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d* 
shared_namedense_37/kernel
s
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes

:d*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
r
Adam/Variable/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/Variable/m
k
#Adam/Variable/m/Read/ReadVariableOpReadVariableOpAdam/Variable/m*
_output_shapes
: *
dtype0
v
Adam/Variable/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/m_1
o
%Adam/Variable/m_1/Read/ReadVariableOpReadVariableOpAdam/Variable/m_1*
_output_shapes
: *
dtype0
?
Adam/dense_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_34/kernel/m
?
*Adam/dense_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_34/bias/m
y
(Adam/dense_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_35/kernel/m
?
*Adam/dense_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/m
y
(Adam/dense_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_36/bias/m
y
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes
:d*
dtype0
?
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_37/kernel/m
?
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes

:d*
dtype0
?
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:*
dtype0
r
Adam/Variable/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/Variable/v
k
#Adam/Variable/v/Read/ReadVariableOpReadVariableOpAdam/Variable/v*
_output_shapes
: *
dtype0
v
Adam/Variable/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/Variable/v_1
o
%Adam/Variable/v_1/Read/ReadVariableOpReadVariableOpAdam/Variable/v_1*
_output_shapes
: *
dtype0
?
Adam/dense_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_34/kernel/v
?
*Adam/dense_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_34/bias/v
y
(Adam/dense_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_34/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_35/kernel/v
?
*Adam/dense_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_35/bias/v
y
(Adam/dense_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_35/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*%
shared_nameAdam/dense_36/bias/v
y
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes
:d*
dtype0
?
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*'
shared_nameAdam/dense_37/kernel/v
?
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes

:d*
dtype0
?
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?3
value?3B?3 B?3
?
c1
c2
encoder
decoder
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
;9
VARIABLE_VALUEVariablec1/.ATTRIBUTES/VARIABLE_VALUE
=;
VARIABLE_VALUE
Variable_1c2/.ATTRIBUTES/VARIABLE_VALUE
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
_build_input_shape
regularization_losses
	variables
trainable_variables
	keras_api
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
	variables
trainable_variables
	keras_api
?
iter

beta_1

beta_2
	decay
learning_ratemambmcmdme mf!mg"mh#mi$mjvkvlvmvnvo vp!vq"vr#vs$vt
 
F
0
1
2
 3
!4
"5
#6
$7
8
9
F
0
1
2
 3
!4
"5
#6
$7
8
9
?
%non_trainable_variables
&metrics
'layer_metrics
regularization_losses
(layer_regularization_losses

)layers
	variables
trainable_variables
 
|
*_inbound_nodes

kernel
bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
|
/_inbound_nodes

kernel
 bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
 
 

0
1
2
 3

0
1
2
 3
?
4non_trainable_variables
5metrics
6layer_metrics
regularization_losses
7layer_regularization_losses

8layers
	variables
trainable_variables
|
9_inbound_nodes

!kernel
"bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
|
>_inbound_nodes

#kernel
$bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
 

!0
"1
#2
$3

!0
"1
#2
$3
?
Cnon_trainable_variables
Dmetrics
Elayer_metrics
regularization_losses
Flayer_regularization_losses

Glayers
	variables
trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_34/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_34/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_35/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_35/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_36/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_36/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_37/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_37/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
 

H0
 
 

0
1
 
 

0
1

0
1
?
Inon_trainable_variables
Jmetrics
Klayer_metrics
+regularization_losses
Llayer_regularization_losses

Mlayers
,	variables
-trainable_variables
 
 

0
 1

0
 1
?
Nnon_trainable_variables
Ometrics
Player_metrics
0regularization_losses
Qlayer_regularization_losses

Rlayers
1	variables
2trainable_variables
 
 
 
 

0
1
 
 

!0
"1

!0
"1
?
Snon_trainable_variables
Tmetrics
Ulayer_metrics
:regularization_losses
Vlayer_regularization_losses

Wlayers
;	variables
<trainable_variables
 
 

#0
$1

#0
$1
?
Xnon_trainable_variables
Ymetrics
Zlayer_metrics
?regularization_losses
[layer_regularization_losses

\layers
@	variables
Atrainable_variables
 
 
 
 

0
1
4
	]total
	^count
_	variables
`	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

]0
^1

_	variables
^\
VARIABLE_VALUEAdam/Variable/m9c1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEAdam/Variable/m_19c2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_34/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_34/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_35/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_35/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_36/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_36/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_37/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_37/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdam/Variable/v9c1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEAdam/Variable/v_19c2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_34/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_34/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_35/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_35/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_36/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_36/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_37/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_37/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_10Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_11Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_12Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_13Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_14Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_15Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_16Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_17Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_18Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_19Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_20Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_21Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_22Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_23Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_24Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_25Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_26Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_27Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_28Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_29Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_30Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_31Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_32Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_33Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_34Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_35Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_36Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_37Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_38Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_39Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_40Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_41Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_42Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_43Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_44Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_45Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_46Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_47Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_48Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_49Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_5Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
{
serving_default_input_50Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_6Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_7Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_8Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_9Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_10serving_default_input_11serving_default_input_12serving_default_input_13serving_default_input_14serving_default_input_15serving_default_input_16serving_default_input_17serving_default_input_18serving_default_input_19serving_default_input_2serving_default_input_20serving_default_input_21serving_default_input_22serving_default_input_23serving_default_input_24serving_default_input_25serving_default_input_26serving_default_input_27serving_default_input_28serving_default_input_29serving_default_input_3serving_default_input_30serving_default_input_31serving_default_input_32serving_default_input_33serving_default_input_34serving_default_input_35serving_default_input_36serving_default_input_37serving_default_input_38serving_default_input_39serving_default_input_4serving_default_input_40serving_default_input_41serving_default_input_42serving_default_input_43serving_default_input_44serving_default_input_45serving_default_input_46serving_default_input_47serving_default_input_48serving_default_input_49serving_default_input_5serving_default_input_50serving_default_input_6serving_default_input_7serving_default_input_8serving_default_input_9dense_34/kerneldense_34/biasdense_35/kerneldense_35/biasVariable
Variable_1dense_36/kerneldense_36/biasdense_37/kerneldense_37/bias*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

23456789:;*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_423136
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOp!dense_35/bias/Read/ReadVariableOp#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp#Adam/Variable/m/Read/ReadVariableOp%Adam/Variable/m_1/Read/ReadVariableOp*Adam/dense_34/kernel/m/Read/ReadVariableOp(Adam/dense_34/bias/m/Read/ReadVariableOp*Adam/dense_35/kernel/m/Read/ReadVariableOp(Adam/dense_35/bias/m/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp#Adam/Variable/v/Read/ReadVariableOp%Adam/Variable/v_1/Read/ReadVariableOp*Adam/dense_34/kernel/v/Read/ReadVariableOp(Adam/dense_34/bias/v/Read/ReadVariableOp*Adam/dense_35/kernel/v/Read/ReadVariableOp(Adam/dense_35/bias/v/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_425017
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable
Variable_1	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense_34/kerneldense_34/biasdense_35/kerneldense_35/biasdense_36/kerneldense_36/biasdense_37/kerneldense_37/biastotalcountAdam/Variable/mAdam/Variable/m_1Adam/dense_34/kernel/mAdam/dense_34/bias/mAdam/dense_35/kernel/mAdam/dense_35/bias/mAdam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/Variable/vAdam/Variable/v_1Adam/dense_34/kernel/vAdam/dense_34/bias/vAdam/dense_35/kernel/vAdam/dense_35/bias/vAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/v*1
Tin*
(2&*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_425138??
?1
?
D__inference_dense_34_layer_call_and_return_conditional_losses_421134

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Selu?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
__inference_loss_fn_5_4247949
5dense_36_bias_regularizer_abs_readvariableop_resource
identity??
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_36_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_36_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1f
IdentityIdentity#dense_36/bias/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?
?
.__inference_sequential_17_layer_call_fn_424341

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4218472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?]
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_421770
dense_36_input
dense_36_421699
dense_36_421701
dense_37_421704
dense_37_421706
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_421699dense_36_421701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_4215622"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_421704dense_37_421706*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4216192"
 dense_37/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421699*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421699*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421701*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421701*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421704*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421704*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421706*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421706*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input
?
?
.__inference_sequential_17_layer_call_fn_421945
dense_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input
?
j
__inference_loss_fn_1_4245549
5dense_34_bias_regularizer_abs_readvariableop_resource
identity??
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_34_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_34_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1f
IdentityIdentity#dense_34/bias/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ݡ
?

G__inference_conjugacy_8_layer_call_and_return_conditional_losses_422516
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
input_9
input_10
input_11
input_12
input_13
input_14
input_15
input_16
input_17
input_18
input_19
input_20
input_21
input_22
input_23
input_24
input_25
input_26
input_27
input_28
input_29
input_30
input_31
input_32
input_33
input_34
input_35
input_36
input_37
input_38
input_39
input_40
input_41
input_42
input_43
input_44
input_45
input_46
input_47
input_48
input_49
input_50
sequential_16_422309
sequential_16_422311
sequential_16_422313
sequential_16_422315
readvariableop_resource
readvariableop_1_resource
sequential_17_422326
sequential_17_422328
sequential_17_422330
sequential_17_422332
identity

identity_1

identity_2

identity_3

identity_4??%sequential_16/StatefulPartitionedCall?'sequential_16/StatefulPartitionedCall_1?'sequential_16/StatefulPartitionedCall_2?%sequential_17/StatefulPartitionedCall?'sequential_17/StatefulPartitionedCall_1?'sequential_17/StatefulPartitionedCall_2?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_422309sequential_16_422311sequential_16_422313sequential_16_422315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062'
%sequential_16/StatefulPartitionedCallp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp?
mulMulReadVariableOp:value:0.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
mul|
SquareSquare.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Squarev
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1m
mul_1MulReadVariableOp_1:value:0
Square:y:0*
T0*'
_output_shapes
:?????????2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCalladd:z:0sequential_17_422326sequential_17_422328sequential_17_422330sequential_17_422332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342'
%sequential_17/StatefulPartitionedCall?
'sequential_16/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_16_422309sequential_16_422311sequential_16_422313sequential_16_422315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062)
'sequential_16/StatefulPartitionedCall_1t
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2?
mul_2MulReadVariableOp_2:value:0.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
Square_1Square.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Square_1v
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_3o
mul_3MulReadVariableOp_3:value:0Square_1:y:0*
T0*'
_output_shapes
:?????????2
mul_3_
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_1?
subSub0sequential_16/StatefulPartitionedCall_1:output:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
subY
Square_2Squaresub:z:0*
T0*'
_output_shapes
:?????????2

Square_2_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstS
MeanMeanSquare_2:y:0Const:output:0*
T0*
_output_shapes
: 2
Mean[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ya
truedivRealDivMean:output:0truediv/y:output:0*
T0*
_output_shapes
: 2	
truediv?
'sequential_17/StatefulPartitionedCall_1StatefulPartitionedCall	add_1:z:0sequential_17_422326sequential_17_422328sequential_17_422330sequential_17_422332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342)
'sequential_17/StatefulPartitionedCall_1?
sub_1Subinput_20sequential_17/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:?????????2
sub_1[
Square_3Square	sub_1:z:0*
T0*'
_output_shapes
:?????????2

Square_3c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1Y
Mean_1MeanSquare_3:y:0Const_1:output:0*
T0*
_output_shapes
: 2
Mean_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yi
	truediv_1RealDivMean_1:output:0truediv_1/y:output:0*
T0*
_output_shapes
: 2
	truediv_1?
'sequential_16/StatefulPartitionedCall_2StatefulPartitionedCallinput_3sequential_16_422309sequential_16_422311sequential_16_422313sequential_16_422315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062)
'sequential_16/StatefulPartitionedCall_2t
ReadVariableOp_4ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_4l
mul_4MulReadVariableOp_4:value:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_4[
Square_4Square	add_1:z:0*
T0*'
_output_shapes
:?????????2

Square_4v
ReadVariableOp_5ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_5o
mul_5MulReadVariableOp_5:value:0Square_4:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_2AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_2?
sub_2Sub0sequential_16/StatefulPartitionedCall_2:output:0	add_2:z:0*
T0*'
_output_shapes
:?????????2
sub_2[
Square_5Square	sub_2:z:0*
T0*'
_output_shapes
:?????????2

Square_5c
Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_2Y
Mean_2MeanSquare_5:y:0Const_2:output:0*
T0*
_output_shapes
: 2
Mean_2_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yi
	truediv_2RealDivMean_2:output:0truediv_2/y:output:0*
T0*
_output_shapes
: 2
	truediv_2?
'sequential_17/StatefulPartitionedCall_2StatefulPartitionedCall	add_2:z:0sequential_17_422326sequential_17_422328sequential_17_422330sequential_17_422332*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342)
'sequential_17/StatefulPartitionedCall_2?
sub_3Subinput_30sequential_17/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:?????????2
sub_3[
Square_6Square	sub_3:z:0*
T0*'
_output_shapes
:?????????2

Square_6c
Const_3Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_3Y
Mean_3MeanSquare_6:y:0Const_3:output:0*
T0*
_output_shapes
: 2
Mean_3_
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_3/yi
	truediv_3RealDivMean_3:output:0truediv_3/y:output:0*
T0*
_output_shapes
: 2
	truediv_3?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422309*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422309*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422311*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422311*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422313*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422313*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422315*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422315*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422326*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422326*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422328*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422328*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422330*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422330*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422332*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422332*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitytruediv:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identitytruediv_1:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identitytruediv_2:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitytruediv_3:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2R
'sequential_16/StatefulPartitionedCall_1'sequential_16/StatefulPartitionedCall_12R
'sequential_16/StatefulPartitionedCall_2'sequential_16/StatefulPartitionedCall_22N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall2R
'sequential_17/StatefulPartitionedCall_1'sequential_17/StatefulPartitionedCall_12R
'sequential_17/StatefulPartitionedCall_2'sequential_17/StatefulPartitionedCall_2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_9:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_12:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_13:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_14:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_15:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_16:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_17:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_18:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_19:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_20:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_21:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_22:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_23:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_24:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_25:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_26:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_27:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_28:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_29:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_30:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_31:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_32:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_33:Q!M
'
_output_shapes
:?????????
"
_user_specified_name
input_34:Q"M
'
_output_shapes
:?????????
"
_user_specified_name
input_35:Q#M
'
_output_shapes
:?????????
"
_user_specified_name
input_36:Q$M
'
_output_shapes
:?????????
"
_user_specified_name
input_37:Q%M
'
_output_shapes
:?????????
"
_user_specified_name
input_38:Q&M
'
_output_shapes
:?????????
"
_user_specified_name
input_39:Q'M
'
_output_shapes
:?????????
"
_user_specified_name
input_40:Q(M
'
_output_shapes
:?????????
"
_user_specified_name
input_41:Q)M
'
_output_shapes
:?????????
"
_user_specified_name
input_42:Q*M
'
_output_shapes
:?????????
"
_user_specified_name
input_43:Q+M
'
_output_shapes
:?????????
"
_user_specified_name
input_44:Q,M
'
_output_shapes
:?????????
"
_user_specified_name
input_45:Q-M
'
_output_shapes
:?????????
"
_user_specified_name
input_46:Q.M
'
_output_shapes
:?????????
"
_user_specified_name
input_47:Q/M
'
_output_shapes
:?????????
"
_user_specified_name
input_48:Q0M
'
_output_shapes
:?????????
"
_user_specified_name
input_49:Q1M
'
_output_shapes
:?????????
"
_user_specified_name
input_50
?
?
.__inference_sequential_16_layer_call_fn_424112

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?b
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_424086

inputs+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMulinputs&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_34/BiasAdds
dense_34/SeluSeludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_34/Selu?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/Selu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdds
dense_35/SeluSeludense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_35/Selu?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1o
IdentityIdentitydense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
D__inference_dense_35_layer_call_and_return_conditional_losses_421191

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Selu?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?1
?
D__inference_dense_36_layer_call_and_return_conditional_losses_421562

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Selu?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_16_layer_call_fn_421517
dense_34_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_34_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_34_input
?b
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_424250

inputs+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulinputs&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_36/BiasAdds
dense_36/SeluSeludense_36/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_36/Selu?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Selu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdds
dense_37/SeluSeludense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Selu?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1o
IdentityIdentitydense_37/Selu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?8
?
$__inference_signature_wrapper_423136
input_1
input_10
input_11
input_12
input_13
input_14
input_15
input_16
input_17
input_18
input_19
input_2
input_20
input_21
input_22
input_23
input_24
input_25
input_26
input_27
input_28
input_29
input_3
input_30
input_31
input_32
input_33
input_34
input_35
input_36
input_37
input_38
input_39
input_4
input_40
input_41
input_42
input_43
input_44
input_45
input_46
input_47
input_48
input_49
input_5
input_50
input_6
input_7
input_8
input_9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7input_8input_9input_10input_11input_12input_13input_14input_15input_16input_17input_18input_19input_20input_21input_22input_23input_24input_25input_26input_27input_28input_29input_30input_31input_32input_33input_34input_35input_36input_37input_38input_39input_40input_41input_42input_43input_44input_45input_46input_47input_48input_49input_50unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

23456789:;*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_4210892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_10:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_11:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_12:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_13:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_14:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_15:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_16:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_17:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
input_18:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
input_19:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_20:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_21:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_22:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_23:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_24:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_25:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_26:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_27:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_28:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_29:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_30:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_31:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_32:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_33:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_34:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_35:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_36:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_37:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_38:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_39:P!L
'
_output_shapes
:?????????
!
_user_specified_name	input_4:Q"M
'
_output_shapes
:?????????
"
_user_specified_name
input_40:Q#M
'
_output_shapes
:?????????
"
_user_specified_name
input_41:Q$M
'
_output_shapes
:?????????
"
_user_specified_name
input_42:Q%M
'
_output_shapes
:?????????
"
_user_specified_name
input_43:Q&M
'
_output_shapes
:?????????
"
_user_specified_name
input_44:Q'M
'
_output_shapes
:?????????
"
_user_specified_name
input_45:Q(M
'
_output_shapes
:?????????
"
_user_specified_name
input_46:Q)M
'
_output_shapes
:?????????
"
_user_specified_name
input_47:Q*M
'
_output_shapes
:?????????
"
_user_specified_name
input_48:Q+M
'
_output_shapes
:?????????
"
_user_specified_name
input_49:P,L
'
_output_shapes
:?????????
!
_user_specified_name	input_5:Q-M
'
_output_shapes
:?????????
"
_user_specified_name
input_50:P.L
'
_output_shapes
:?????????
!
_user_specified_name	input_6:P/L
'
_output_shapes
:?????????
!
_user_specified_name	input_7:P0L
'
_output_shapes
:?????????
!
_user_specified_name	input_8:P1L
'
_output_shapes
:?????????
!
_user_specified_name	input_9
?1
?
D__inference_dense_36_layer_call_and_return_conditional_losses_424665

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Selu?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?
,__inference_conjugacy_8_layer_call_fn_422932
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
input_9
input_10
input_11
input_12
input_13
input_14
input_15
input_16
input_17
input_18
input_19
input_20
input_21
input_22
input_23
input_24
input_25
input_26
input_27
input_28
input_29
input_30
input_31
input_32
input_33
input_34
input_35
input_36
input_37
input_38
input_39
input_40
input_41
input_42
input_43
input_44
input_45
input_46
input_47
input_48
input_49
input_50
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7input_8input_9input_10input_11input_12input_13input_14input_15input_16input_17input_18input_19input_20input_21input_22input_23input_24input_25input_26input_27input_28input_29input_30input_31input_32input_33input_34input_35input_36input_37input_38input_39input_40input_41input_42input_43input_44input_45input_46input_47input_48input_49input_50unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*G
Tin@
>2<*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????: : : : *,
_read_only_resource_inputs

23456789:;*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_4228272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_9:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_12:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_13:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_14:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_15:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_16:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_17:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_18:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_19:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_20:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_21:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_22:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_23:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_24:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_25:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_26:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_27:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_28:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_29:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_30:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_31:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_32:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_33:Q!M
'
_output_shapes
:?????????
"
_user_specified_name
input_34:Q"M
'
_output_shapes
:?????????
"
_user_specified_name
input_35:Q#M
'
_output_shapes
:?????????
"
_user_specified_name
input_36:Q$M
'
_output_shapes
:?????????
"
_user_specified_name
input_37:Q%M
'
_output_shapes
:?????????
"
_user_specified_name
input_38:Q&M
'
_output_shapes
:?????????
"
_user_specified_name
input_39:Q'M
'
_output_shapes
:?????????
"
_user_specified_name
input_40:Q(M
'
_output_shapes
:?????????
"
_user_specified_name
input_41:Q)M
'
_output_shapes
:?????????
"
_user_specified_name
input_42:Q*M
'
_output_shapes
:?????????
"
_user_specified_name
input_43:Q+M
'
_output_shapes
:?????????
"
_user_specified_name
input_44:Q,M
'
_output_shapes
:?????????
"
_user_specified_name
input_45:Q-M
'
_output_shapes
:?????????
"
_user_specified_name
input_46:Q.M
'
_output_shapes
:?????????
"
_user_specified_name
input_47:Q/M
'
_output_shapes
:?????????
"
_user_specified_name
input_48:Q0M
'
_output_shapes
:?????????
"
_user_specified_name
input_49:Q1M
'
_output_shapes
:?????????
"
_user_specified_name
input_50
?
j
__inference_loss_fn_7_4248349
5dense_37_bias_regularizer_abs_readvariableop_resource
identity??
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_37_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_37_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1f
IdentityIdentity#dense_37/bias/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
??
?	
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_422827
x
x_1
x_2
x_3
x_4
x_5
x_6
x_7
x_8
x_9
x_10
x_11
x_12
x_13
x_14
x_15
x_16
x_17
x_18
x_19
x_20
x_21
x_22
x_23
x_24
x_25
x_26
x_27
x_28
x_29
x_30
x_31
x_32
x_33
x_34
x_35
x_36
x_37
x_38
x_39
x_40
x_41
x_42
x_43
x_44
x_45
x_46
x_47
x_48
x_49
sequential_16_422620
sequential_16_422622
sequential_16_422624
sequential_16_422626
readvariableop_resource
readvariableop_1_resource
sequential_17_422637
sequential_17_422639
sequential_17_422641
sequential_17_422643
identity

identity_1

identity_2

identity_3

identity_4??%sequential_16/StatefulPartitionedCall?'sequential_16/StatefulPartitionedCall_1?'sequential_16/StatefulPartitionedCall_2?%sequential_17/StatefulPartitionedCall?'sequential_17/StatefulPartitionedCall_1?'sequential_17/StatefulPartitionedCall_2?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallxsequential_16_422620sequential_16_422622sequential_16_422624sequential_16_422626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062'
%sequential_16/StatefulPartitionedCallp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp?
mulMulReadVariableOp:value:0.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
mul|
SquareSquare.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Squarev
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1m
mul_1MulReadVariableOp_1:value:0
Square:y:0*
T0*'
_output_shapes
:?????????2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCalladd:z:0sequential_17_422637sequential_17_422639sequential_17_422641sequential_17_422643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342'
%sequential_17/StatefulPartitionedCall?
'sequential_16/StatefulPartitionedCall_1StatefulPartitionedCallx_1sequential_16_422620sequential_16_422622sequential_16_422624sequential_16_422626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062)
'sequential_16/StatefulPartitionedCall_1t
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2?
mul_2MulReadVariableOp_2:value:0.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
Square_1Square.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Square_1v
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_3o
mul_3MulReadVariableOp_3:value:0Square_1:y:0*
T0*'
_output_shapes
:?????????2
mul_3_
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_1?
subSub0sequential_16/StatefulPartitionedCall_1:output:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
subY
Square_2Squaresub:z:0*
T0*'
_output_shapes
:?????????2

Square_2_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstS
MeanMeanSquare_2:y:0Const:output:0*
T0*
_output_shapes
: 2
Mean[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ya
truedivRealDivMean:output:0truediv/y:output:0*
T0*
_output_shapes
: 2	
truediv?
'sequential_17/StatefulPartitionedCall_1StatefulPartitionedCall	add_1:z:0sequential_17_422637sequential_17_422639sequential_17_422641sequential_17_422643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342)
'sequential_17/StatefulPartitionedCall_1~
sub_1Subx_10sequential_17/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:?????????2
sub_1[
Square_3Square	sub_1:z:0*
T0*'
_output_shapes
:?????????2

Square_3c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1Y
Mean_1MeanSquare_3:y:0Const_1:output:0*
T0*
_output_shapes
: 2
Mean_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yi
	truediv_1RealDivMean_1:output:0truediv_1/y:output:0*
T0*
_output_shapes
: 2
	truediv_1?
'sequential_16/StatefulPartitionedCall_2StatefulPartitionedCallx_2sequential_16_422620sequential_16_422622sequential_16_422624sequential_16_422626*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4215062)
'sequential_16/StatefulPartitionedCall_2t
ReadVariableOp_4ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_4l
mul_4MulReadVariableOp_4:value:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_4[
Square_4Square	add_1:z:0*
T0*'
_output_shapes
:?????????2

Square_4v
ReadVariableOp_5ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_5o
mul_5MulReadVariableOp_5:value:0Square_4:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_2AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_2?
sub_2Sub0sequential_16/StatefulPartitionedCall_2:output:0	add_2:z:0*
T0*'
_output_shapes
:?????????2
sub_2[
Square_5Square	sub_2:z:0*
T0*'
_output_shapes
:?????????2

Square_5c
Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_2Y
Mean_2MeanSquare_5:y:0Const_2:output:0*
T0*
_output_shapes
: 2
Mean_2_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yi
	truediv_2RealDivMean_2:output:0truediv_2/y:output:0*
T0*
_output_shapes
: 2
	truediv_2?
'sequential_17/StatefulPartitionedCall_2StatefulPartitionedCall	add_2:z:0sequential_17_422637sequential_17_422639sequential_17_422641sequential_17_422643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342)
'sequential_17/StatefulPartitionedCall_2~
sub_3Subx_20sequential_17/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:?????????2
sub_3[
Square_6Square	sub_3:z:0*
T0*'
_output_shapes
:?????????2

Square_6c
Const_3Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_3Y
Mean_3MeanSquare_6:y:0Const_3:output:0*
T0*
_output_shapes
: 2
Mean_3_
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_3/yi
	truediv_3RealDivMean_3:output:0truediv_3/y:output:0*
T0*
_output_shapes
: 2
	truediv_3?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422620*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422620*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422622*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422622*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422624*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422624*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422626*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422626*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422637*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422637*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422639*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422639*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422641*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422641*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422643*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422643*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitytruediv:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identitytruediv_1:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identitytruediv_2:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitytruediv_3:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2R
'sequential_16/StatefulPartitionedCall_1'sequential_16/StatefulPartitionedCall_12R
'sequential_16/StatefulPartitionedCall_2'sequential_16/StatefulPartitionedCall_22N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall2R
'sequential_17/StatefulPartitionedCall_1'sequential_17/StatefulPartitionedCall_12R
'sequential_17/StatefulPartitionedCall_2'sequential_17/StatefulPartitionedCall_2:J F
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:J	F
'
_output_shapes
:?????????

_user_specified_namex:J
F
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:JF
'
_output_shapes
:?????????

_user_specified_namex:J F
'
_output_shapes
:?????????

_user_specified_namex:J!F
'
_output_shapes
:?????????

_user_specified_namex:J"F
'
_output_shapes
:?????????

_user_specified_namex:J#F
'
_output_shapes
:?????????

_user_specified_namex:J$F
'
_output_shapes
:?????????

_user_specified_namex:J%F
'
_output_shapes
:?????????

_user_specified_namex:J&F
'
_output_shapes
:?????????

_user_specified_namex:J'F
'
_output_shapes
:?????????

_user_specified_namex:J(F
'
_output_shapes
:?????????

_user_specified_namex:J)F
'
_output_shapes
:?????????

_user_specified_namex:J*F
'
_output_shapes
:?????????

_user_specified_namex:J+F
'
_output_shapes
:?????????

_user_specified_namex:J,F
'
_output_shapes
:?????????

_user_specified_namex:J-F
'
_output_shapes
:?????????

_user_specified_namex:J.F
'
_output_shapes
:?????????

_user_specified_namex:J/F
'
_output_shapes
:?????????

_user_specified_namex:J0F
'
_output_shapes
:?????????

_user_specified_namex:J1F
'
_output_shapes
:?????????

_user_specified_namex
?]
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_421506

inputs
dense_34_421435
dense_34_421437
dense_35_421440
dense_35_421442
identity?? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCallinputsdense_34_421435dense_34_421437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4211342"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_421440dense_35_421442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4211912"
 dense_35/StatefulPartitionedCall?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421435*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421435*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421437*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421437*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421440*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421440*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421442*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421442*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?	
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_423425
x_0
x_1
x_2
x_3
x_4
x_5
x_6
x_7
x_8
x_9
x_10
x_11
x_12
x_13
x_14
x_15
x_16
x_17
x_18
x_19
x_20
x_21
x_22
x_23
x_24
x_25
x_26
x_27
x_28
x_29
x_30
x_31
x_32
x_33
x_34
x_35
x_36
x_37
x_38
x_39
x_40
x_41
x_42
x_43
x_44
x_45
x_46
x_47
x_48
x_499
5sequential_16_dense_34_matmul_readvariableop_resource:
6sequential_16_dense_34_biasadd_readvariableop_resource9
5sequential_16_dense_35_matmul_readvariableop_resource:
6sequential_16_dense_35_biasadd_readvariableop_resource
readvariableop_resource
readvariableop_1_resource9
5sequential_17_dense_36_matmul_readvariableop_resource:
6sequential_17_dense_36_biasadd_readvariableop_resource9
5sequential_17_dense_37_matmul_readvariableop_resource:
6sequential_17_dense_37_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4??
,sequential_16/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_16/dense_34/MatMul/ReadVariableOp?
sequential_16/dense_34/MatMulMatMulx_04sequential_16/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/MatMul?
-sequential_16/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_16/dense_34/BiasAdd/ReadVariableOp?
sequential_16/dense_34/BiasAddBiasAdd'sequential_16/dense_34/MatMul:product:05sequential_16/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_16/dense_34/BiasAdd?
sequential_16/dense_34/SeluSelu'sequential_16/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/Selu?
,sequential_16/dense_35/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_16/dense_35/MatMul/ReadVariableOp?
sequential_16/dense_35/MatMulMatMul)sequential_16/dense_34/Selu:activations:04sequential_16/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/MatMul?
-sequential_16/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_16/dense_35/BiasAdd/ReadVariableOp?
sequential_16/dense_35/BiasAddBiasAdd'sequential_16/dense_35/MatMul:product:05sequential_16/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_16/dense_35/BiasAdd?
sequential_16/dense_35/SeluSelu'sequential_16/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/Selup
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp?
mulMulReadVariableOp:value:0)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
mulw
SquareSquare)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
Squarev
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1m
mul_1MulReadVariableOp_1:value:0
Square:y:0*
T0*'
_output_shapes
:?????????2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add?
,sequential_17/dense_36/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_17/dense_36/MatMul/ReadVariableOp?
sequential_17/dense_36/MatMulMatMuladd:z:04sequential_17/dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/MatMul?
-sequential_17/dense_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_17/dense_36/BiasAdd/ReadVariableOp?
sequential_17/dense_36/BiasAddBiasAdd'sequential_17/dense_36/MatMul:product:05sequential_17/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_17/dense_36/BiasAdd?
sequential_17/dense_36/SeluSelu'sequential_17/dense_36/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/Selu?
,sequential_17/dense_37/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_17/dense_37/MatMul/ReadVariableOp?
sequential_17/dense_37/MatMulMatMul)sequential_17/dense_36/Selu:activations:04sequential_17/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/MatMul?
-sequential_17/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_17/dense_37/BiasAdd/ReadVariableOp?
sequential_17/dense_37/BiasAddBiasAdd'sequential_17/dense_37/MatMul:product:05sequential_17/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_17/dense_37/BiasAdd?
sequential_17/dense_37/SeluSelu'sequential_17/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/Selu?
.sequential_16/dense_34/MatMul_1/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_34/MatMul_1/ReadVariableOp?
sequential_16/dense_34/MatMul_1MatMulx_16sequential_16/dense_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_16/dense_34/MatMul_1?
/sequential_16/dense_34/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_16/dense_34/BiasAdd_1/ReadVariableOp?
 sequential_16/dense_34/BiasAdd_1BiasAdd)sequential_16/dense_34/MatMul_1:product:07sequential_16/dense_34/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_16/dense_34/BiasAdd_1?
sequential_16/dense_34/Selu_1Selu)sequential_16/dense_34/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/Selu_1?
.sequential_16/dense_35/MatMul_1/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_35/MatMul_1/ReadVariableOp?
sequential_16/dense_35/MatMul_1MatMul+sequential_16/dense_34/Selu_1:activations:06sequential_16/dense_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_16/dense_35/MatMul_1?
/sequential_16/dense_35/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_16/dense_35/BiasAdd_1/ReadVariableOp?
 sequential_16/dense_35/BiasAdd_1BiasAdd)sequential_16/dense_35/MatMul_1:product:07sequential_16/dense_35/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_16/dense_35/BiasAdd_1?
sequential_16/dense_35/Selu_1Selu)sequential_16/dense_35/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/Selu_1t
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2?
mul_2MulReadVariableOp_2:value:0)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
mul_2{
Square_1Square)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2

Square_1v
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_3o
mul_3MulReadVariableOp_3:value:0Square_1:y:0*
T0*'
_output_shapes
:?????????2
mul_3_
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_1{
subSub+sequential_16/dense_35/Selu_1:activations:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
subY
Square_2Squaresub:z:0*
T0*'
_output_shapes
:?????????2

Square_2_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstS
MeanMeanSquare_2:y:0Const:output:0*
T0*
_output_shapes
: 2
Mean[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ya
truedivRealDivMean:output:0truediv/y:output:0*
T0*
_output_shapes
: 2	
truediv?
.sequential_17/dense_36/MatMul_1/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_36/MatMul_1/ReadVariableOp?
sequential_17/dense_36/MatMul_1MatMul	add_1:z:06sequential_17/dense_36/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_17/dense_36/MatMul_1?
/sequential_17/dense_36/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_17/dense_36/BiasAdd_1/ReadVariableOp?
 sequential_17/dense_36/BiasAdd_1BiasAdd)sequential_17/dense_36/MatMul_1:product:07sequential_17/dense_36/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_17/dense_36/BiasAdd_1?
sequential_17/dense_36/Selu_1Selu)sequential_17/dense_36/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/Selu_1?
.sequential_17/dense_37/MatMul_1/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_37/MatMul_1/ReadVariableOp?
sequential_17/dense_37/MatMul_1MatMul+sequential_17/dense_36/Selu_1:activations:06sequential_17/dense_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_17/dense_37/MatMul_1?
/sequential_17/dense_37/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_17/dense_37/BiasAdd_1/ReadVariableOp?
 sequential_17/dense_37/BiasAdd_1BiasAdd)sequential_17/dense_37/MatMul_1:product:07sequential_17/dense_37/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_17/dense_37/BiasAdd_1?
sequential_17/dense_37/Selu_1Selu)sequential_17/dense_37/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/Selu_1y
sub_1Subx_1+sequential_17/dense_37/Selu_1:activations:0*
T0*'
_output_shapes
:?????????2
sub_1[
Square_3Square	sub_1:z:0*
T0*'
_output_shapes
:?????????2

Square_3c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1Y
Mean_1MeanSquare_3:y:0Const_1:output:0*
T0*
_output_shapes
: 2
Mean_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yi
	truediv_1RealDivMean_1:output:0truediv_1/y:output:0*
T0*
_output_shapes
: 2
	truediv_1?
.sequential_16/dense_34/MatMul_2/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_34/MatMul_2/ReadVariableOp?
sequential_16/dense_34/MatMul_2MatMulx_26sequential_16/dense_34/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_16/dense_34/MatMul_2?
/sequential_16/dense_34/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_16/dense_34/BiasAdd_2/ReadVariableOp?
 sequential_16/dense_34/BiasAdd_2BiasAdd)sequential_16/dense_34/MatMul_2:product:07sequential_16/dense_34/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_16/dense_34/BiasAdd_2?
sequential_16/dense_34/Selu_2Selu)sequential_16/dense_34/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/Selu_2?
.sequential_16/dense_35/MatMul_2/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_35/MatMul_2/ReadVariableOp?
sequential_16/dense_35/MatMul_2MatMul+sequential_16/dense_34/Selu_2:activations:06sequential_16/dense_35/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_16/dense_35/MatMul_2?
/sequential_16/dense_35/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_16/dense_35/BiasAdd_2/ReadVariableOp?
 sequential_16/dense_35/BiasAdd_2BiasAdd)sequential_16/dense_35/MatMul_2:product:07sequential_16/dense_35/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_16/dense_35/BiasAdd_2?
sequential_16/dense_35/Selu_2Selu)sequential_16/dense_35/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/Selu_2t
ReadVariableOp_4ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_4l
mul_4MulReadVariableOp_4:value:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_4[
Square_4Square	add_1:z:0*
T0*'
_output_shapes
:?????????2

Square_4v
ReadVariableOp_5ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_5o
mul_5MulReadVariableOp_5:value:0Square_4:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_2AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_2
sub_2Sub+sequential_16/dense_35/Selu_2:activations:0	add_2:z:0*
T0*'
_output_shapes
:?????????2
sub_2[
Square_5Square	sub_2:z:0*
T0*'
_output_shapes
:?????????2

Square_5c
Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_2Y
Mean_2MeanSquare_5:y:0Const_2:output:0*
T0*
_output_shapes
: 2
Mean_2_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yi
	truediv_2RealDivMean_2:output:0truediv_2/y:output:0*
T0*
_output_shapes
: 2
	truediv_2?
.sequential_17/dense_36/MatMul_2/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_36/MatMul_2/ReadVariableOp?
sequential_17/dense_36/MatMul_2MatMul	add_2:z:06sequential_17/dense_36/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_17/dense_36/MatMul_2?
/sequential_17/dense_36/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_17/dense_36/BiasAdd_2/ReadVariableOp?
 sequential_17/dense_36/BiasAdd_2BiasAdd)sequential_17/dense_36/MatMul_2:product:07sequential_17/dense_36/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_17/dense_36/BiasAdd_2?
sequential_17/dense_36/Selu_2Selu)sequential_17/dense_36/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/Selu_2?
.sequential_17/dense_37/MatMul_2/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_37/MatMul_2/ReadVariableOp?
sequential_17/dense_37/MatMul_2MatMul+sequential_17/dense_36/Selu_2:activations:06sequential_17/dense_37/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_17/dense_37/MatMul_2?
/sequential_17/dense_37/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_17/dense_37/BiasAdd_2/ReadVariableOp?
 sequential_17/dense_37/BiasAdd_2BiasAdd)sequential_17/dense_37/MatMul_2:product:07sequential_17/dense_37/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_17/dense_37/BiasAdd_2?
sequential_17/dense_37/Selu_2Selu)sequential_17/dense_37/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/Selu_2y
sub_3Subx_2+sequential_17/dense_37/Selu_2:activations:0*
T0*'
_output_shapes
:?????????2
sub_3[
Square_6Square	sub_3:z:0*
T0*'
_output_shapes
:?????????2

Square_6c
Const_3Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_3Y
Mean_3MeanSquare_6:y:0Const_3:output:0*
T0*
_output_shapes
: 2
Mean_3_
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_3/yi
	truediv_3RealDivMean_3:output:0truediv_3/y:output:0*
T0*
_output_shapes
: 2
	truediv_3?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1}
IdentityIdentity)sequential_17/dense_37/Selu:activations:0*
T0*'
_output_shapes
:?????????2

IdentityR

Identity_1Identitytruediv:z:0*
T0*
_output_shapes
: 2

Identity_1T

Identity_2Identitytruediv_1:z:0*
T0*
_output_shapes
: 2

Identity_2T

Identity_3Identitytruediv_2:z:0*
T0*
_output_shapes
: 2

Identity_3T

Identity_4Identitytruediv_3:z:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:::::::::::L H
'
_output_shapes
:?????????

_user_specified_namex/0:LH
'
_output_shapes
:?????????

_user_specified_namex/1:LH
'
_output_shapes
:?????????

_user_specified_namex/2:LH
'
_output_shapes
:?????????

_user_specified_namex/3:LH
'
_output_shapes
:?????????

_user_specified_namex/4:LH
'
_output_shapes
:?????????

_user_specified_namex/5:LH
'
_output_shapes
:?????????

_user_specified_namex/6:LH
'
_output_shapes
:?????????

_user_specified_namex/7:LH
'
_output_shapes
:?????????

_user_specified_namex/8:L	H
'
_output_shapes
:?????????

_user_specified_namex/9:M
I
'
_output_shapes
:?????????

_user_specified_namex/10:MI
'
_output_shapes
:?????????

_user_specified_namex/11:MI
'
_output_shapes
:?????????

_user_specified_namex/12:MI
'
_output_shapes
:?????????

_user_specified_namex/13:MI
'
_output_shapes
:?????????

_user_specified_namex/14:MI
'
_output_shapes
:?????????

_user_specified_namex/15:MI
'
_output_shapes
:?????????

_user_specified_namex/16:MI
'
_output_shapes
:?????????

_user_specified_namex/17:MI
'
_output_shapes
:?????????

_user_specified_namex/18:MI
'
_output_shapes
:?????????

_user_specified_namex/19:MI
'
_output_shapes
:?????????

_user_specified_namex/20:MI
'
_output_shapes
:?????????

_user_specified_namex/21:MI
'
_output_shapes
:?????????

_user_specified_namex/22:MI
'
_output_shapes
:?????????

_user_specified_namex/23:MI
'
_output_shapes
:?????????

_user_specified_namex/24:MI
'
_output_shapes
:?????????

_user_specified_namex/25:MI
'
_output_shapes
:?????????

_user_specified_namex/26:MI
'
_output_shapes
:?????????

_user_specified_namex/27:MI
'
_output_shapes
:?????????

_user_specified_namex/28:MI
'
_output_shapes
:?????????

_user_specified_namex/29:MI
'
_output_shapes
:?????????

_user_specified_namex/30:MI
'
_output_shapes
:?????????

_user_specified_namex/31:M I
'
_output_shapes
:?????????

_user_specified_namex/32:M!I
'
_output_shapes
:?????????

_user_specified_namex/33:M"I
'
_output_shapes
:?????????

_user_specified_namex/34:M#I
'
_output_shapes
:?????????

_user_specified_namex/35:M$I
'
_output_shapes
:?????????

_user_specified_namex/36:M%I
'
_output_shapes
:?????????

_user_specified_namex/37:M&I
'
_output_shapes
:?????????

_user_specified_namex/38:M'I
'
_output_shapes
:?????????

_user_specified_namex/39:M(I
'
_output_shapes
:?????????

_user_specified_namex/40:M)I
'
_output_shapes
:?????????

_user_specified_namex/41:M*I
'
_output_shapes
:?????????

_user_specified_namex/42:M+I
'
_output_shapes
:?????????

_user_specified_namex/43:M,I
'
_output_shapes
:?????????

_user_specified_namex/44:M-I
'
_output_shapes
:?????????

_user_specified_namex/45:M.I
'
_output_shapes
:?????????

_user_specified_namex/46:M/I
'
_output_shapes
:?????????

_user_specified_namex/47:M0I
'
_output_shapes
:?????????

_user_specified_namex/48:M1I
'
_output_shapes
:?????????

_user_specified_namex/49
?
?
.__inference_sequential_16_layer_call_fn_424099

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4214192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?]
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_421419

inputs
dense_34_421348
dense_34_421350
dense_35_421353
dense_35_421355
identity?? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCallinputsdense_34_421348dense_34_421350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4211342"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_421353dense_35_421355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4211912"
 dense_35/StatefulPartitionedCall?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421348*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421348*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421350*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421350*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421353*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421353*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421355*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421355*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_16_layer_call_fn_421430
dense_34_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_34_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4214192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_34_input
?9
?
,__inference_conjugacy_8_layer_call_fn_422854
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
input_9
input_10
input_11
input_12
input_13
input_14
input_15
input_16
input_17
input_18
input_19
input_20
input_21
input_22
input_23
input_24
input_25
input_26
input_27
input_28
input_29
input_30
input_31
input_32
input_33
input_34
input_35
input_36
input_37
input_38
input_39
input_40
input_41
input_42
input_43
input_44
input_45
input_46
input_47
input_48
input_49
input_50
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7input_8input_9input_10input_11input_12input_13input_14input_15input_16input_17input_18input_19input_20input_21input_22input_23input_24input_25input_26input_27input_28input_29input_30input_31input_32input_33input_34input_35input_36input_37input_38input_39input_40input_41input_42input_43input_44input_45input_46input_47input_48input_49input_50unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*G
Tin@
>2<*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????: : : : *,
_read_only_resource_inputs

23456789:;*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_4228272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_9:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_12:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_13:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_14:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_15:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_16:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_17:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_18:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_19:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_20:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_21:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_22:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_23:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_24:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_25:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_26:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_27:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_28:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_29:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_30:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_31:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_32:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_33:Q!M
'
_output_shapes
:?????????
"
_user_specified_name
input_34:Q"M
'
_output_shapes
:?????????
"
_user_specified_name
input_35:Q#M
'
_output_shapes
:?????????
"
_user_specified_name
input_36:Q$M
'
_output_shapes
:?????????
"
_user_specified_name
input_37:Q%M
'
_output_shapes
:?????????
"
_user_specified_name
input_38:Q&M
'
_output_shapes
:?????????
"
_user_specified_name
input_39:Q'M
'
_output_shapes
:?????????
"
_user_specified_name
input_40:Q(M
'
_output_shapes
:?????????
"
_user_specified_name
input_41:Q)M
'
_output_shapes
:?????????
"
_user_specified_name
input_42:Q*M
'
_output_shapes
:?????????
"
_user_specified_name
input_43:Q+M
'
_output_shapes
:?????????
"
_user_specified_name
input_44:Q,M
'
_output_shapes
:?????????
"
_user_specified_name
input_45:Q-M
'
_output_shapes
:?????????
"
_user_specified_name
input_46:Q.M
'
_output_shapes
:?????????
"
_user_specified_name
input_47:Q/M
'
_output_shapes
:?????????
"
_user_specified_name
input_48:Q0M
'
_output_shapes
:?????????
"
_user_specified_name
input_49:Q1M
'
_output_shapes
:?????????
"
_user_specified_name
input_50
?]
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_421847

inputs
dense_36_421776
dense_36_421778
dense_37_421781
dense_37_421783
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_421776dense_36_421778*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_4215622"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_421781dense_37_421783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4216192"
 dense_37/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421776*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421776*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421778*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421778*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421781*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421781*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421783*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421783*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
__inference_loss_fn_0_424534;
7dense_34_kernel_regularizer_abs_readvariableop_resource
identity??
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_34_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_34_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1h
IdentityIdentity%dense_34/kernel/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?
~
)__inference_dense_35_layer_call_fn_424514

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4211912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
Ę
?
"__inference__traced_restore_425138
file_prefix
assignvariableop_variable!
assignvariableop_1_variable_1 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate&
"assignvariableop_7_dense_34_kernel$
 assignvariableop_8_dense_34_bias&
"assignvariableop_9_dense_35_kernel%
!assignvariableop_10_dense_35_bias'
#assignvariableop_11_dense_36_kernel%
!assignvariableop_12_dense_36_bias'
#assignvariableop_13_dense_37_kernel%
!assignvariableop_14_dense_37_bias
assignvariableop_15_total
assignvariableop_16_count'
#assignvariableop_17_adam_variable_m)
%assignvariableop_18_adam_variable_m_1.
*assignvariableop_19_adam_dense_34_kernel_m,
(assignvariableop_20_adam_dense_34_bias_m.
*assignvariableop_21_adam_dense_35_kernel_m,
(assignvariableop_22_adam_dense_35_bias_m.
*assignvariableop_23_adam_dense_36_kernel_m,
(assignvariableop_24_adam_dense_36_bias_m.
*assignvariableop_25_adam_dense_37_kernel_m,
(assignvariableop_26_adam_dense_37_bias_m'
#assignvariableop_27_adam_variable_v)
%assignvariableop_28_adam_variable_v_1.
*assignvariableop_29_adam_dense_34_kernel_v,
(assignvariableop_30_adam_dense_34_bias_v.
*assignvariableop_31_adam_dense_35_kernel_v,
(assignvariableop_32_adam_dense_35_bias_v.
*assignvariableop_33_adam_dense_36_kernel_v,
(assignvariableop_34_adam_dense_36_bias_v.
*assignvariableop_35_adam_dense_37_kernel_v,
(assignvariableop_36_adam_dense_37_bias_v
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&Bc1/.ATTRIBUTES/VARIABLE_VALUEBc2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB9c1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9c2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9c1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9c2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_34_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_dense_34_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_35_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp!assignvariableop_10_dense_35_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_36_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense_36_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_37_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense_37_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_adam_variable_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp%assignvariableop_18_adam_variable_m_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_34_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_34_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_35_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_35_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_36_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_36_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_37_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_37_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp#assignvariableop_27_adam_variable_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_variable_v_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_34_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_34_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_35_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_35_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_dense_36_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_dense_36_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_37_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_37_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
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
?
?
.__inference_sequential_17_layer_call_fn_421858
dense_36_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4218472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input
?
~
)__inference_dense_34_layer_call_fn_424434

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4211342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_dense_36_layer_call_fn_424674

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_4215622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
l
__inference_loss_fn_4_424774;
7dense_36_kernel_regularizer_abs_readvariableop_resource
identity??
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_36_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_36_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1h
IdentityIdentity%dense_36/kernel/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?4
?
,__inference_conjugacy_8_layer_call_fn_423792
x_0
x_1
x_2
x_3
x_4
x_5
x_6
x_7
x_8
x_9
x_10
x_11
x_12
x_13
x_14
x_15
x_16
x_17
x_18
x_19
x_20
x_21
x_22
x_23
x_24
x_25
x_26
x_27
x_28
x_29
x_30
x_31
x_32
x_33
x_34
x_35
x_36
x_37
x_38
x_39
x_40
x_41
x_42
x_43
x_44
x_45
x_46
x_47
x_48
x_49
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallx_0x_1x_2x_3x_4x_5x_6x_7x_8x_9x_10x_11x_12x_13x_14x_15x_16x_17x_18x_19x_20x_21x_22x_23x_24x_25x_26x_27x_28x_29x_30x_31x_32x_33x_34x_35x_36x_37x_38x_39x_40x_41x_42x_43x_44x_45x_46x_47x_48x_49unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*G
Tin@
>2<*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????: : : : *,
_read_only_resource_inputs

23456789:;*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_4228272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_namex/0:LH
'
_output_shapes
:?????????

_user_specified_namex/1:LH
'
_output_shapes
:?????????

_user_specified_namex/2:LH
'
_output_shapes
:?????????

_user_specified_namex/3:LH
'
_output_shapes
:?????????

_user_specified_namex/4:LH
'
_output_shapes
:?????????

_user_specified_namex/5:LH
'
_output_shapes
:?????????

_user_specified_namex/6:LH
'
_output_shapes
:?????????

_user_specified_namex/7:LH
'
_output_shapes
:?????????

_user_specified_namex/8:L	H
'
_output_shapes
:?????????

_user_specified_namex/9:M
I
'
_output_shapes
:?????????

_user_specified_namex/10:MI
'
_output_shapes
:?????????

_user_specified_namex/11:MI
'
_output_shapes
:?????????

_user_specified_namex/12:MI
'
_output_shapes
:?????????

_user_specified_namex/13:MI
'
_output_shapes
:?????????

_user_specified_namex/14:MI
'
_output_shapes
:?????????

_user_specified_namex/15:MI
'
_output_shapes
:?????????

_user_specified_namex/16:MI
'
_output_shapes
:?????????

_user_specified_namex/17:MI
'
_output_shapes
:?????????

_user_specified_namex/18:MI
'
_output_shapes
:?????????

_user_specified_namex/19:MI
'
_output_shapes
:?????????

_user_specified_namex/20:MI
'
_output_shapes
:?????????

_user_specified_namex/21:MI
'
_output_shapes
:?????????

_user_specified_namex/22:MI
'
_output_shapes
:?????????

_user_specified_namex/23:MI
'
_output_shapes
:?????????

_user_specified_namex/24:MI
'
_output_shapes
:?????????

_user_specified_namex/25:MI
'
_output_shapes
:?????????

_user_specified_namex/26:MI
'
_output_shapes
:?????????

_user_specified_namex/27:MI
'
_output_shapes
:?????????

_user_specified_namex/28:MI
'
_output_shapes
:?????????

_user_specified_namex/29:MI
'
_output_shapes
:?????????

_user_specified_namex/30:MI
'
_output_shapes
:?????????

_user_specified_namex/31:M I
'
_output_shapes
:?????????

_user_specified_namex/32:M!I
'
_output_shapes
:?????????

_user_specified_namex/33:M"I
'
_output_shapes
:?????????

_user_specified_namex/34:M#I
'
_output_shapes
:?????????

_user_specified_namex/35:M$I
'
_output_shapes
:?????????

_user_specified_namex/36:M%I
'
_output_shapes
:?????????

_user_specified_namex/37:M&I
'
_output_shapes
:?????????

_user_specified_namex/38:M'I
'
_output_shapes
:?????????

_user_specified_namex/39:M(I
'
_output_shapes
:?????????

_user_specified_namex/40:M)I
'
_output_shapes
:?????????

_user_specified_namex/41:M*I
'
_output_shapes
:?????????

_user_specified_namex/42:M+I
'
_output_shapes
:?????????

_user_specified_namex/43:M,I
'
_output_shapes
:?????????

_user_specified_namex/44:M-I
'
_output_shapes
:?????????

_user_specified_namex/45:M.I
'
_output_shapes
:?????????

_user_specified_namex/46:M/I
'
_output_shapes
:?????????

_user_specified_namex/47:M0I
'
_output_shapes
:?????????

_user_specified_namex/48:M1I
'
_output_shapes
:?????????

_user_specified_namex/49
?
~
)__inference_dense_37_layer_call_fn_424754

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4216192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
ݡ
?

G__inference_conjugacy_8_layer_call_and_return_conditional_losses_422257
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
input_9
input_10
input_11
input_12
input_13
input_14
input_15
input_16
input_17
input_18
input_19
input_20
input_21
input_22
input_23
input_24
input_25
input_26
input_27
input_28
input_29
input_30
input_31
input_32
input_33
input_34
input_35
input_36
input_37
input_38
input_39
input_40
input_41
input_42
input_43
input_44
input_45
input_46
input_47
input_48
input_49
input_50
sequential_16_422024
sequential_16_422026
sequential_16_422028
sequential_16_422030
readvariableop_resource
readvariableop_1_resource
sequential_17_422067
sequential_17_422069
sequential_17_422071
sequential_17_422073
identity

identity_1

identity_2

identity_3

identity_4??%sequential_16/StatefulPartitionedCall?'sequential_16/StatefulPartitionedCall_1?'sequential_16/StatefulPartitionedCall_2?%sequential_17/StatefulPartitionedCall?'sequential_17/StatefulPartitionedCall_1?'sequential_17/StatefulPartitionedCall_2?
%sequential_16/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_16_422024sequential_16_422026sequential_16_422028sequential_16_422030*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4214192'
%sequential_16/StatefulPartitionedCallp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp?
mulMulReadVariableOp:value:0.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
mul|
SquareSquare.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
Squarev
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1m
mul_1MulReadVariableOp_1:value:0
Square:y:0*
T0*'
_output_shapes
:?????????2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add?
%sequential_17/StatefulPartitionedCallStatefulPartitionedCalladd:z:0sequential_17_422067sequential_17_422069sequential_17_422071sequential_17_422073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4218472'
%sequential_17/StatefulPartitionedCall?
'sequential_16/StatefulPartitionedCall_1StatefulPartitionedCallinput_2sequential_16_422024sequential_16_422026sequential_16_422028sequential_16_422030*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4214192)
'sequential_16/StatefulPartitionedCall_1t
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2?
mul_2MulReadVariableOp_2:value:0.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
mul_2?
Square_1Square.sequential_16/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Square_1v
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_3o
mul_3MulReadVariableOp_3:value:0Square_1:y:0*
T0*'
_output_shapes
:?????????2
mul_3_
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_1?
subSub0sequential_16/StatefulPartitionedCall_1:output:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
subY
Square_2Squaresub:z:0*
T0*'
_output_shapes
:?????????2

Square_2_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstS
MeanMeanSquare_2:y:0Const:output:0*
T0*
_output_shapes
: 2
Mean[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ya
truedivRealDivMean:output:0truediv/y:output:0*
T0*
_output_shapes
: 2	
truediv?
'sequential_17/StatefulPartitionedCall_1StatefulPartitionedCall	add_1:z:0sequential_17_422067sequential_17_422069sequential_17_422071sequential_17_422073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4218472)
'sequential_17/StatefulPartitionedCall_1?
sub_1Subinput_20sequential_17/StatefulPartitionedCall_1:output:0*
T0*'
_output_shapes
:?????????2
sub_1[
Square_3Square	sub_1:z:0*
T0*'
_output_shapes
:?????????2

Square_3c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1Y
Mean_1MeanSquare_3:y:0Const_1:output:0*
T0*
_output_shapes
: 2
Mean_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yi
	truediv_1RealDivMean_1:output:0truediv_1/y:output:0*
T0*
_output_shapes
: 2
	truediv_1?
'sequential_16/StatefulPartitionedCall_2StatefulPartitionedCallinput_3sequential_16_422024sequential_16_422026sequential_16_422028sequential_16_422030*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_16_layer_call_and_return_conditional_losses_4214192)
'sequential_16/StatefulPartitionedCall_2t
ReadVariableOp_4ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_4l
mul_4MulReadVariableOp_4:value:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_4[
Square_4Square	add_1:z:0*
T0*'
_output_shapes
:?????????2

Square_4v
ReadVariableOp_5ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_5o
mul_5MulReadVariableOp_5:value:0Square_4:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_2AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_2?
sub_2Sub0sequential_16/StatefulPartitionedCall_2:output:0	add_2:z:0*
T0*'
_output_shapes
:?????????2
sub_2[
Square_5Square	sub_2:z:0*
T0*'
_output_shapes
:?????????2

Square_5c
Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_2Y
Mean_2MeanSquare_5:y:0Const_2:output:0*
T0*
_output_shapes
: 2
Mean_2_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yi
	truediv_2RealDivMean_2:output:0truediv_2/y:output:0*
T0*
_output_shapes
: 2
	truediv_2?
'sequential_17/StatefulPartitionedCall_2StatefulPartitionedCall	add_2:z:0sequential_17_422067sequential_17_422069sequential_17_422071sequential_17_422073*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4218472)
'sequential_17/StatefulPartitionedCall_2?
sub_3Subinput_30sequential_17/StatefulPartitionedCall_2:output:0*
T0*'
_output_shapes
:?????????2
sub_3[
Square_6Square	sub_3:z:0*
T0*'
_output_shapes
:?????????2

Square_6c
Const_3Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_3Y
Mean_3MeanSquare_6:y:0Const_3:output:0*
T0*
_output_shapes
: 2
Mean_3_
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_3/yi
	truediv_3RealDivMean_3:output:0truediv_3/y:output:0*
T0*
_output_shapes
: 2
	truediv_3?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422024*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422024*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422026*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422026*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422028*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422028*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_16_422030*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_16_422030*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422067*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422067*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422069*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422069*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422071*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422071*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpsequential_17_422073*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpsequential_17_422073*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1?
IdentityIdentity.sequential_17/StatefulPartitionedCall:output:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitytruediv:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2Identitytruediv_1:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_2?

Identity_3Identitytruediv_2:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_3?

Identity_4Identitytruediv_3:z:0&^sequential_16/StatefulPartitionedCall(^sequential_16/StatefulPartitionedCall_1(^sequential_16/StatefulPartitionedCall_2&^sequential_17/StatefulPartitionedCall(^sequential_17/StatefulPartitionedCall_1(^sequential_17/StatefulPartitionedCall_2*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::2N
%sequential_16/StatefulPartitionedCall%sequential_16/StatefulPartitionedCall2R
'sequential_16/StatefulPartitionedCall_1'sequential_16/StatefulPartitionedCall_12R
'sequential_16/StatefulPartitionedCall_2'sequential_16/StatefulPartitionedCall_22N
%sequential_17/StatefulPartitionedCall%sequential_17/StatefulPartitionedCall2R
'sequential_17/StatefulPartitionedCall_1'sequential_17/StatefulPartitionedCall_12R
'sequential_17/StatefulPartitionedCall_2'sequential_17/StatefulPartitionedCall_2:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_9:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_12:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_13:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_14:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_15:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_16:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_17:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_18:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_19:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_20:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_21:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_22:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_23:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_24:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_25:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_26:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_27:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_28:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_29:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_30:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_31:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_32:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_33:Q!M
'
_output_shapes
:?????????
"
_user_specified_name
input_34:Q"M
'
_output_shapes
:?????????
"
_user_specified_name
input_35:Q#M
'
_output_shapes
:?????????
"
_user_specified_name
input_36:Q$M
'
_output_shapes
:?????????
"
_user_specified_name
input_37:Q%M
'
_output_shapes
:?????????
"
_user_specified_name
input_38:Q&M
'
_output_shapes
:?????????
"
_user_specified_name
input_39:Q'M
'
_output_shapes
:?????????
"
_user_specified_name
input_40:Q(M
'
_output_shapes
:?????????
"
_user_specified_name
input_41:Q)M
'
_output_shapes
:?????????
"
_user_specified_name
input_42:Q*M
'
_output_shapes
:?????????
"
_user_specified_name
input_43:Q+M
'
_output_shapes
:?????????
"
_user_specified_name
input_44:Q,M
'
_output_shapes
:?????????
"
_user_specified_name
input_45:Q-M
'
_output_shapes
:?????????
"
_user_specified_name
input_46:Q.M
'
_output_shapes
:?????????
"
_user_specified_name
input_47:Q/M
'
_output_shapes
:?????????
"
_user_specified_name
input_48:Q0M
'
_output_shapes
:?????????
"
_user_specified_name
input_49:Q1M
'
_output_shapes
:?????????
"
_user_specified_name
input_50
?J
?
__inference__traced_save_425017
file_prefix'
#savev2_variable_read_readvariableop)
%savev2_variable_1_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop,
(savev2_dense_35_bias_read_readvariableop.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop.
*savev2_adam_variable_m_read_readvariableop0
,savev2_adam_variable_m_1_read_readvariableop5
1savev2_adam_dense_34_kernel_m_read_readvariableop3
/savev2_adam_dense_34_bias_m_read_readvariableop5
1savev2_adam_dense_35_kernel_m_read_readvariableop3
/savev2_adam_dense_35_bias_m_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop.
*savev2_adam_variable_v_read_readvariableop0
,savev2_adam_variable_v_1_read_readvariableop5
1savev2_adam_dense_34_kernel_v_read_readvariableop3
/savev2_adam_dense_34_bias_v_read_readvariableop5
1savev2_adam_dense_35_kernel_v_read_readvariableop3
/savev2_adam_dense_35_bias_v_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_760c8a8949a8447799f2526695f87b88/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&Bc1/.ATTRIBUTES/VARIABLE_VALUEBc2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB9c1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9c2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB9c1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB9c2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableop*savev2_dense_35_kernel_read_readvariableop(savev2_dense_35_bias_read_readvariableop*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop*savev2_adam_variable_m_read_readvariableop,savev2_adam_variable_m_1_read_readvariableop1savev2_adam_dense_34_kernel_m_read_readvariableop/savev2_adam_dense_34_bias_m_read_readvariableop1savev2_adam_dense_35_kernel_m_read_readvariableop/savev2_adam_dense_35_bias_m_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop*savev2_adam_variable_v_read_readvariableop,savev2_adam_variable_v_1_read_readvariableop1savev2_adam_dense_34_kernel_v_read_readvariableop/savev2_adam_dense_34_bias_v_read_readvariableop1savev2_adam_dense_35_kernel_v_read_readvariableop/savev2_adam_dense_35_bias_v_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : :d:d:d::d:d:d:: : : : :d:d:d::d:d:d:: : :d:d:d::d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 	

_output_shapes
:d:$
 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
:d:$  

_output_shapes

:d: !

_output_shapes
::$" 

_output_shapes

:d: #

_output_shapes
:d:$$ 

_output_shapes

:d: %

_output_shapes
::&

_output_shapes
: 
?]
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_421268
dense_34_input
dense_34_421145
dense_34_421147
dense_35_421202
dense_35_421204
identity?? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCalldense_34_inputdense_34_421145dense_34_421147*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4211342"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_421202dense_35_421204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4211912"
 dense_35/StatefulPartitionedCall?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421145*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421145*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421147*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421147*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421202*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421202*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421204*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421204*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_34_input
?
l
__inference_loss_fn_6_424814;
7dense_37_kernel_regularizer_abs_readvariableop_resource
identity??
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_37_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_37_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1h
IdentityIdentity%dense_37/kernel/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?
l
__inference_loss_fn_2_424574;
7dense_35_kernel_regularizer_abs_readvariableop_resource
identity??
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_35_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_35_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1h
IdentityIdentity%dense_35/kernel/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?]
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_421696
dense_36_input
dense_36_421573
dense_36_421575
dense_37_421630
dense_37_421632
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_421573dense_36_421575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_4215622"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_421630dense_37_421632*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4216192"
 dense_37/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421573*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421573*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421575*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421575*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421630*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421630*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421632*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421632*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_36_input
??
?

!__inference__wrapped_model_421089
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
input_9
input_10
input_11
input_12
input_13
input_14
input_15
input_16
input_17
input_18
input_19
input_20
input_21
input_22
input_23
input_24
input_25
input_26
input_27
input_28
input_29
input_30
input_31
input_32
input_33
input_34
input_35
input_36
input_37
input_38
input_39
input_40
input_41
input_42
input_43
input_44
input_45
input_46
input_47
input_48
input_49
input_50E
Aconjugacy_8_sequential_16_dense_34_matmul_readvariableop_resourceF
Bconjugacy_8_sequential_16_dense_34_biasadd_readvariableop_resourceE
Aconjugacy_8_sequential_16_dense_35_matmul_readvariableop_resourceF
Bconjugacy_8_sequential_16_dense_35_biasadd_readvariableop_resource'
#conjugacy_8_readvariableop_resource)
%conjugacy_8_readvariableop_1_resourceE
Aconjugacy_8_sequential_17_dense_36_matmul_readvariableop_resourceF
Bconjugacy_8_sequential_17_dense_36_biasadd_readvariableop_resourceE
Aconjugacy_8_sequential_17_dense_37_matmul_readvariableop_resourceF
Bconjugacy_8_sequential_17_dense_37_biasadd_readvariableop_resource
identity??
8conjugacy_8/sequential_16/dense_34/MatMul/ReadVariableOpReadVariableOpAconjugacy_8_sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02:
8conjugacy_8/sequential_16/dense_34/MatMul/ReadVariableOp?
)conjugacy_8/sequential_16/dense_34/MatMulMatMulinput_1@conjugacy_8/sequential_16/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2+
)conjugacy_8/sequential_16/dense_34/MatMul?
9conjugacy_8/sequential_16/dense_34/BiasAdd/ReadVariableOpReadVariableOpBconjugacy_8_sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02;
9conjugacy_8/sequential_16/dense_34/BiasAdd/ReadVariableOp?
*conjugacy_8/sequential_16/dense_34/BiasAddBiasAdd3conjugacy_8/sequential_16/dense_34/MatMul:product:0Aconjugacy_8/sequential_16/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2,
*conjugacy_8/sequential_16/dense_34/BiasAdd?
'conjugacy_8/sequential_16/dense_34/SeluSelu3conjugacy_8/sequential_16/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2)
'conjugacy_8/sequential_16/dense_34/Selu?
8conjugacy_8/sequential_16/dense_35/MatMul/ReadVariableOpReadVariableOpAconjugacy_8_sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02:
8conjugacy_8/sequential_16/dense_35/MatMul/ReadVariableOp?
)conjugacy_8/sequential_16/dense_35/MatMulMatMul5conjugacy_8/sequential_16/dense_34/Selu:activations:0@conjugacy_8/sequential_16/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)conjugacy_8/sequential_16/dense_35/MatMul?
9conjugacy_8/sequential_16/dense_35/BiasAdd/ReadVariableOpReadVariableOpBconjugacy_8_sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9conjugacy_8/sequential_16/dense_35/BiasAdd/ReadVariableOp?
*conjugacy_8/sequential_16/dense_35/BiasAddBiasAdd3conjugacy_8/sequential_16/dense_35/MatMul:product:0Aconjugacy_8/sequential_16/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*conjugacy_8/sequential_16/dense_35/BiasAdd?
'conjugacy_8/sequential_16/dense_35/SeluSelu3conjugacy_8/sequential_16/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2)
'conjugacy_8/sequential_16/dense_35/Selu?
conjugacy_8/ReadVariableOpReadVariableOp#conjugacy_8_readvariableop_resource*
_output_shapes
: *
dtype02
conjugacy_8/ReadVariableOp?
conjugacy_8/mulMul"conjugacy_8/ReadVariableOp:value:05conjugacy_8/sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/mul?
conjugacy_8/SquareSquare5conjugacy_8/sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/Square?
conjugacy_8/ReadVariableOp_1ReadVariableOp%conjugacy_8_readvariableop_1_resource*
_output_shapes
: *
dtype02
conjugacy_8/ReadVariableOp_1?
conjugacy_8/mul_1Mul$conjugacy_8/ReadVariableOp_1:value:0conjugacy_8/Square:y:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/mul_1?
conjugacy_8/addAddV2conjugacy_8/mul:z:0conjugacy_8/mul_1:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/add?
8conjugacy_8/sequential_17/dense_36/MatMul/ReadVariableOpReadVariableOpAconjugacy_8_sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02:
8conjugacy_8/sequential_17/dense_36/MatMul/ReadVariableOp?
)conjugacy_8/sequential_17/dense_36/MatMulMatMulconjugacy_8/add:z:0@conjugacy_8/sequential_17/dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2+
)conjugacy_8/sequential_17/dense_36/MatMul?
9conjugacy_8/sequential_17/dense_36/BiasAdd/ReadVariableOpReadVariableOpBconjugacy_8_sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02;
9conjugacy_8/sequential_17/dense_36/BiasAdd/ReadVariableOp?
*conjugacy_8/sequential_17/dense_36/BiasAddBiasAdd3conjugacy_8/sequential_17/dense_36/MatMul:product:0Aconjugacy_8/sequential_17/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2,
*conjugacy_8/sequential_17/dense_36/BiasAdd?
'conjugacy_8/sequential_17/dense_36/SeluSelu3conjugacy_8/sequential_17/dense_36/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2)
'conjugacy_8/sequential_17/dense_36/Selu?
8conjugacy_8/sequential_17/dense_37/MatMul/ReadVariableOpReadVariableOpAconjugacy_8_sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02:
8conjugacy_8/sequential_17/dense_37/MatMul/ReadVariableOp?
)conjugacy_8/sequential_17/dense_37/MatMulMatMul5conjugacy_8/sequential_17/dense_36/Selu:activations:0@conjugacy_8/sequential_17/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2+
)conjugacy_8/sequential_17/dense_37/MatMul?
9conjugacy_8/sequential_17/dense_37/BiasAdd/ReadVariableOpReadVariableOpBconjugacy_8_sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9conjugacy_8/sequential_17/dense_37/BiasAdd/ReadVariableOp?
*conjugacy_8/sequential_17/dense_37/BiasAddBiasAdd3conjugacy_8/sequential_17/dense_37/MatMul:product:0Aconjugacy_8/sequential_17/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*conjugacy_8/sequential_17/dense_37/BiasAdd?
'conjugacy_8/sequential_17/dense_37/SeluSelu3conjugacy_8/sequential_17/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2)
'conjugacy_8/sequential_17/dense_37/Selu?
:conjugacy_8/sequential_16/dense_34/MatMul_1/ReadVariableOpReadVariableOpAconjugacy_8_sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_16/dense_34/MatMul_1/ReadVariableOp?
+conjugacy_8/sequential_16/dense_34/MatMul_1MatMulinput_2Bconjugacy_8/sequential_16/dense_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+conjugacy_8/sequential_16/dense_34/MatMul_1?
;conjugacy_8/sequential_16/dense_34/BiasAdd_1/ReadVariableOpReadVariableOpBconjugacy_8_sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;conjugacy_8/sequential_16/dense_34/BiasAdd_1/ReadVariableOp?
,conjugacy_8/sequential_16/dense_34/BiasAdd_1BiasAdd5conjugacy_8/sequential_16/dense_34/MatMul_1:product:0Cconjugacy_8/sequential_16/dense_34/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,conjugacy_8/sequential_16/dense_34/BiasAdd_1?
)conjugacy_8/sequential_16/dense_34/Selu_1Selu5conjugacy_8/sequential_16/dense_34/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????d2+
)conjugacy_8/sequential_16/dense_34/Selu_1?
:conjugacy_8/sequential_16/dense_35/MatMul_1/ReadVariableOpReadVariableOpAconjugacy_8_sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_16/dense_35/MatMul_1/ReadVariableOp?
+conjugacy_8/sequential_16/dense_35/MatMul_1MatMul7conjugacy_8/sequential_16/dense_34/Selu_1:activations:0Bconjugacy_8/sequential_16/dense_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+conjugacy_8/sequential_16/dense_35/MatMul_1?
;conjugacy_8/sequential_16/dense_35/BiasAdd_1/ReadVariableOpReadVariableOpBconjugacy_8_sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;conjugacy_8/sequential_16/dense_35/BiasAdd_1/ReadVariableOp?
,conjugacy_8/sequential_16/dense_35/BiasAdd_1BiasAdd5conjugacy_8/sequential_16/dense_35/MatMul_1:product:0Cconjugacy_8/sequential_16/dense_35/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,conjugacy_8/sequential_16/dense_35/BiasAdd_1?
)conjugacy_8/sequential_16/dense_35/Selu_1Selu5conjugacy_8/sequential_16/dense_35/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2+
)conjugacy_8/sequential_16/dense_35/Selu_1?
conjugacy_8/ReadVariableOp_2ReadVariableOp#conjugacy_8_readvariableop_resource*
_output_shapes
: *
dtype02
conjugacy_8/ReadVariableOp_2?
conjugacy_8/mul_2Mul$conjugacy_8/ReadVariableOp_2:value:05conjugacy_8/sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/mul_2?
conjugacy_8/Square_1Square5conjugacy_8/sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/Square_1?
conjugacy_8/ReadVariableOp_3ReadVariableOp%conjugacy_8_readvariableop_1_resource*
_output_shapes
: *
dtype02
conjugacy_8/ReadVariableOp_3?
conjugacy_8/mul_3Mul$conjugacy_8/ReadVariableOp_3:value:0conjugacy_8/Square_1:y:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/mul_3?
conjugacy_8/add_1AddV2conjugacy_8/mul_2:z:0conjugacy_8/mul_3:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/add_1?
conjugacy_8/subSub7conjugacy_8/sequential_16/dense_35/Selu_1:activations:0conjugacy_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/sub}
conjugacy_8/Square_2Squareconjugacy_8/sub:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/Square_2w
conjugacy_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
conjugacy_8/Const?
conjugacy_8/MeanMeanconjugacy_8/Square_2:y:0conjugacy_8/Const:output:0*
T0*
_output_shapes
: 2
conjugacy_8/Means
conjugacy_8/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conjugacy_8/truediv/y?
conjugacy_8/truedivRealDivconjugacy_8/Mean:output:0conjugacy_8/truediv/y:output:0*
T0*
_output_shapes
: 2
conjugacy_8/truediv?
:conjugacy_8/sequential_17/dense_36/MatMul_1/ReadVariableOpReadVariableOpAconjugacy_8_sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_17/dense_36/MatMul_1/ReadVariableOp?
+conjugacy_8/sequential_17/dense_36/MatMul_1MatMulconjugacy_8/add_1:z:0Bconjugacy_8/sequential_17/dense_36/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+conjugacy_8/sequential_17/dense_36/MatMul_1?
;conjugacy_8/sequential_17/dense_36/BiasAdd_1/ReadVariableOpReadVariableOpBconjugacy_8_sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;conjugacy_8/sequential_17/dense_36/BiasAdd_1/ReadVariableOp?
,conjugacy_8/sequential_17/dense_36/BiasAdd_1BiasAdd5conjugacy_8/sequential_17/dense_36/MatMul_1:product:0Cconjugacy_8/sequential_17/dense_36/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,conjugacy_8/sequential_17/dense_36/BiasAdd_1?
)conjugacy_8/sequential_17/dense_36/Selu_1Selu5conjugacy_8/sequential_17/dense_36/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????d2+
)conjugacy_8/sequential_17/dense_36/Selu_1?
:conjugacy_8/sequential_17/dense_37/MatMul_1/ReadVariableOpReadVariableOpAconjugacy_8_sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_17/dense_37/MatMul_1/ReadVariableOp?
+conjugacy_8/sequential_17/dense_37/MatMul_1MatMul7conjugacy_8/sequential_17/dense_36/Selu_1:activations:0Bconjugacy_8/sequential_17/dense_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+conjugacy_8/sequential_17/dense_37/MatMul_1?
;conjugacy_8/sequential_17/dense_37/BiasAdd_1/ReadVariableOpReadVariableOpBconjugacy_8_sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;conjugacy_8/sequential_17/dense_37/BiasAdd_1/ReadVariableOp?
,conjugacy_8/sequential_17/dense_37/BiasAdd_1BiasAdd5conjugacy_8/sequential_17/dense_37/MatMul_1:product:0Cconjugacy_8/sequential_17/dense_37/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,conjugacy_8/sequential_17/dense_37/BiasAdd_1?
)conjugacy_8/sequential_17/dense_37/Selu_1Selu5conjugacy_8/sequential_17/dense_37/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2+
)conjugacy_8/sequential_17/dense_37/Selu_1?
conjugacy_8/sub_1Subinput_27conjugacy_8/sequential_17/dense_37/Selu_1:activations:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/sub_1
conjugacy_8/Square_3Squareconjugacy_8/sub_1:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/Square_3{
conjugacy_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
conjugacy_8/Const_1?
conjugacy_8/Mean_1Meanconjugacy_8/Square_3:y:0conjugacy_8/Const_1:output:0*
T0*
_output_shapes
: 2
conjugacy_8/Mean_1w
conjugacy_8/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conjugacy_8/truediv_1/y?
conjugacy_8/truediv_1RealDivconjugacy_8/Mean_1:output:0 conjugacy_8/truediv_1/y:output:0*
T0*
_output_shapes
: 2
conjugacy_8/truediv_1?
:conjugacy_8/sequential_16/dense_34/MatMul_2/ReadVariableOpReadVariableOpAconjugacy_8_sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_16/dense_34/MatMul_2/ReadVariableOp?
+conjugacy_8/sequential_16/dense_34/MatMul_2MatMulinput_3Bconjugacy_8/sequential_16/dense_34/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+conjugacy_8/sequential_16/dense_34/MatMul_2?
;conjugacy_8/sequential_16/dense_34/BiasAdd_2/ReadVariableOpReadVariableOpBconjugacy_8_sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;conjugacy_8/sequential_16/dense_34/BiasAdd_2/ReadVariableOp?
,conjugacy_8/sequential_16/dense_34/BiasAdd_2BiasAdd5conjugacy_8/sequential_16/dense_34/MatMul_2:product:0Cconjugacy_8/sequential_16/dense_34/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,conjugacy_8/sequential_16/dense_34/BiasAdd_2?
)conjugacy_8/sequential_16/dense_34/Selu_2Selu5conjugacy_8/sequential_16/dense_34/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????d2+
)conjugacy_8/sequential_16/dense_34/Selu_2?
:conjugacy_8/sequential_16/dense_35/MatMul_2/ReadVariableOpReadVariableOpAconjugacy_8_sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_16/dense_35/MatMul_2/ReadVariableOp?
+conjugacy_8/sequential_16/dense_35/MatMul_2MatMul7conjugacy_8/sequential_16/dense_34/Selu_2:activations:0Bconjugacy_8/sequential_16/dense_35/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+conjugacy_8/sequential_16/dense_35/MatMul_2?
;conjugacy_8/sequential_16/dense_35/BiasAdd_2/ReadVariableOpReadVariableOpBconjugacy_8_sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;conjugacy_8/sequential_16/dense_35/BiasAdd_2/ReadVariableOp?
,conjugacy_8/sequential_16/dense_35/BiasAdd_2BiasAdd5conjugacy_8/sequential_16/dense_35/MatMul_2:product:0Cconjugacy_8/sequential_16/dense_35/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,conjugacy_8/sequential_16/dense_35/BiasAdd_2?
)conjugacy_8/sequential_16/dense_35/Selu_2Selu5conjugacy_8/sequential_16/dense_35/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????2+
)conjugacy_8/sequential_16/dense_35/Selu_2?
conjugacy_8/ReadVariableOp_4ReadVariableOp#conjugacy_8_readvariableop_resource*
_output_shapes
: *
dtype02
conjugacy_8/ReadVariableOp_4?
conjugacy_8/mul_4Mul$conjugacy_8/ReadVariableOp_4:value:0conjugacy_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/mul_4
conjugacy_8/Square_4Squareconjugacy_8/add_1:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/Square_4?
conjugacy_8/ReadVariableOp_5ReadVariableOp%conjugacy_8_readvariableop_1_resource*
_output_shapes
: *
dtype02
conjugacy_8/ReadVariableOp_5?
conjugacy_8/mul_5Mul$conjugacy_8/ReadVariableOp_5:value:0conjugacy_8/Square_4:y:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/mul_5?
conjugacy_8/add_2AddV2conjugacy_8/mul_4:z:0conjugacy_8/mul_5:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/add_2?
conjugacy_8/sub_2Sub7conjugacy_8/sequential_16/dense_35/Selu_2:activations:0conjugacy_8/add_2:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/sub_2
conjugacy_8/Square_5Squareconjugacy_8/sub_2:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/Square_5{
conjugacy_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2
conjugacy_8/Const_2?
conjugacy_8/Mean_2Meanconjugacy_8/Square_5:y:0conjugacy_8/Const_2:output:0*
T0*
_output_shapes
: 2
conjugacy_8/Mean_2w
conjugacy_8/truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conjugacy_8/truediv_2/y?
conjugacy_8/truediv_2RealDivconjugacy_8/Mean_2:output:0 conjugacy_8/truediv_2/y:output:0*
T0*
_output_shapes
: 2
conjugacy_8/truediv_2?
:conjugacy_8/sequential_17/dense_36/MatMul_2/ReadVariableOpReadVariableOpAconjugacy_8_sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_17/dense_36/MatMul_2/ReadVariableOp?
+conjugacy_8/sequential_17/dense_36/MatMul_2MatMulconjugacy_8/add_2:z:0Bconjugacy_8/sequential_17/dense_36/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2-
+conjugacy_8/sequential_17/dense_36/MatMul_2?
;conjugacy_8/sequential_17/dense_36/BiasAdd_2/ReadVariableOpReadVariableOpBconjugacy_8_sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02=
;conjugacy_8/sequential_17/dense_36/BiasAdd_2/ReadVariableOp?
,conjugacy_8/sequential_17/dense_36/BiasAdd_2BiasAdd5conjugacy_8/sequential_17/dense_36/MatMul_2:product:0Cconjugacy_8/sequential_17/dense_36/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2.
,conjugacy_8/sequential_17/dense_36/BiasAdd_2?
)conjugacy_8/sequential_17/dense_36/Selu_2Selu5conjugacy_8/sequential_17/dense_36/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????d2+
)conjugacy_8/sequential_17/dense_36/Selu_2?
:conjugacy_8/sequential_17/dense_37/MatMul_2/ReadVariableOpReadVariableOpAconjugacy_8_sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02<
:conjugacy_8/sequential_17/dense_37/MatMul_2/ReadVariableOp?
+conjugacy_8/sequential_17/dense_37/MatMul_2MatMul7conjugacy_8/sequential_17/dense_36/Selu_2:activations:0Bconjugacy_8/sequential_17/dense_37/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+conjugacy_8/sequential_17/dense_37/MatMul_2?
;conjugacy_8/sequential_17/dense_37/BiasAdd_2/ReadVariableOpReadVariableOpBconjugacy_8_sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;conjugacy_8/sequential_17/dense_37/BiasAdd_2/ReadVariableOp?
,conjugacy_8/sequential_17/dense_37/BiasAdd_2BiasAdd5conjugacy_8/sequential_17/dense_37/MatMul_2:product:0Cconjugacy_8/sequential_17/dense_37/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2.
,conjugacy_8/sequential_17/dense_37/BiasAdd_2?
)conjugacy_8/sequential_17/dense_37/Selu_2Selu5conjugacy_8/sequential_17/dense_37/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????2+
)conjugacy_8/sequential_17/dense_37/Selu_2?
conjugacy_8/sub_3Subinput_37conjugacy_8/sequential_17/dense_37/Selu_2:activations:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/sub_3
conjugacy_8/Square_6Squareconjugacy_8/sub_3:z:0*
T0*'
_output_shapes
:?????????2
conjugacy_8/Square_6{
conjugacy_8/Const_3Const*
_output_shapes
:*
dtype0*
valueB"       2
conjugacy_8/Const_3?
conjugacy_8/Mean_3Meanconjugacy_8/Square_6:y:0conjugacy_8/Const_3:output:0*
T0*
_output_shapes
: 2
conjugacy_8/Mean_3w
conjugacy_8/truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
conjugacy_8/truediv_3/y?
conjugacy_8/truediv_3RealDivconjugacy_8/Mean_3:output:0 conjugacy_8/truediv_3/y:output:0*
T0*
_output_shapes
: 2
conjugacy_8/truediv_3?
IdentityIdentity5conjugacy_8/sequential_17/dense_37/Selu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:::::::::::P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_5:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_6:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_7:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_8:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_9:Q	M
'
_output_shapes
:?????????
"
_user_specified_name
input_10:Q
M
'
_output_shapes
:?????????
"
_user_specified_name
input_11:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_12:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_13:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_14:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_15:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_16:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_17:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_18:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_19:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_20:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_21:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_22:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_23:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_24:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_25:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_26:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_27:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_28:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_29:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_30:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_31:QM
'
_output_shapes
:?????????
"
_user_specified_name
input_32:Q M
'
_output_shapes
:?????????
"
_user_specified_name
input_33:Q!M
'
_output_shapes
:?????????
"
_user_specified_name
input_34:Q"M
'
_output_shapes
:?????????
"
_user_specified_name
input_35:Q#M
'
_output_shapes
:?????????
"
_user_specified_name
input_36:Q$M
'
_output_shapes
:?????????
"
_user_specified_name
input_37:Q%M
'
_output_shapes
:?????????
"
_user_specified_name
input_38:Q&M
'
_output_shapes
:?????????
"
_user_specified_name
input_39:Q'M
'
_output_shapes
:?????????
"
_user_specified_name
input_40:Q(M
'
_output_shapes
:?????????
"
_user_specified_name
input_41:Q)M
'
_output_shapes
:?????????
"
_user_specified_name
input_42:Q*M
'
_output_shapes
:?????????
"
_user_specified_name
input_43:Q+M
'
_output_shapes
:?????????
"
_user_specified_name
input_44:Q,M
'
_output_shapes
:?????????
"
_user_specified_name
input_45:Q-M
'
_output_shapes
:?????????
"
_user_specified_name
input_46:Q.M
'
_output_shapes
:?????????
"
_user_specified_name
input_47:Q/M
'
_output_shapes
:?????????
"
_user_specified_name
input_48:Q0M
'
_output_shapes
:?????????
"
_user_specified_name
input_49:Q1M
'
_output_shapes
:?????????
"
_user_specified_name
input_50
?]
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_421342
dense_34_input
dense_34_421271
dense_34_421273
dense_35_421276
dense_35_421278
identity?? dense_34/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
 dense_34/StatefulPartitionedCallStatefulPartitionedCalldense_34_inputdense_34_421271dense_34_421273*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_34_layer_call_and_return_conditional_losses_4211342"
 dense_34/StatefulPartitionedCall?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall)dense_34/StatefulPartitionedCall:output:0dense_35_421276dense_35_421278*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_35_layer_call_and_return_conditional_losses_4211912"
 dense_35/StatefulPartitionedCall?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421271*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421271*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_34_421273*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_34_421273*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421276*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421276*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_35_421278*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_35_421278*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0!^dense_34/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
'
_output_shapes
:?????????
(
_user_specified_namedense_34_input
?1
?
D__inference_dense_37_layer_call_and_return_conditional_losses_424745

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Selu?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?1
?
D__inference_dense_35_layer_call_and_return_conditional_losses_424505

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Selu?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?b
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_424328

inputs+
'dense_36_matmul_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource+
'dense_37_matmul_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??
dense_36/MatMul/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_36/MatMul/ReadVariableOp?
dense_36/MatMulMatMulinputs&dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_36/MatMul?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/MatMul:product:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_36/BiasAdds
dense_36/SeluSeludense_36/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_36/Selu?
dense_37/MatMul/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_37/MatMul/ReadVariableOp?
dense_37/MatMulMatMuldense_36/Selu:activations:0&dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/MatMul?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/MatMul:product:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/BiasAdds
dense_37/SeluSeludense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_37/Selu?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1o
IdentityIdentitydense_37/Selu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
D__inference_dense_34_layer_call_and_return_conditional_losses_424425

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
Selu?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????d2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?1
?
D__inference_dense_37_layer_call_and_return_conditional_losses_421619

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
SeluSeluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Selu?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1f
IdentityIdentitySelu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????d:::O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
.__inference_sequential_17_layer_call_fn_424354

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_4219342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?4
?
,__inference_conjugacy_8_layer_call_fn_423870
x_0
x_1
x_2
x_3
x_4
x_5
x_6
x_7
x_8
x_9
x_10
x_11
x_12
x_13
x_14
x_15
x_16
x_17
x_18
x_19
x_20
x_21
x_22
x_23
x_24
x_25
x_26
x_27
x_28
x_29
x_30
x_31
x_32
x_33
x_34
x_35
x_36
x_37
x_38
x_39
x_40
x_41
x_42
x_43
x_44
x_45
x_46
x_47
x_48
x_49
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallx_0x_1x_2x_3x_4x_5x_6x_7x_8x_9x_10x_11x_12x_13x_14x_15x_16x_17x_18x_19x_20x_21x_22x_23x_24x_25x_26x_27x_28x_29x_30x_31x_32x_33x_34x_35x_36x_37x_38x_39x_40x_41x_42x_43x_44x_45x_46x_47x_48x_49unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*G
Tin@
>2<*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????: : : : *,
_read_only_resource_inputs

23456789:;*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_4228272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_namex/0:LH
'
_output_shapes
:?????????

_user_specified_namex/1:LH
'
_output_shapes
:?????????

_user_specified_namex/2:LH
'
_output_shapes
:?????????

_user_specified_namex/3:LH
'
_output_shapes
:?????????

_user_specified_namex/4:LH
'
_output_shapes
:?????????

_user_specified_namex/5:LH
'
_output_shapes
:?????????

_user_specified_namex/6:LH
'
_output_shapes
:?????????

_user_specified_namex/7:LH
'
_output_shapes
:?????????

_user_specified_namex/8:L	H
'
_output_shapes
:?????????

_user_specified_namex/9:M
I
'
_output_shapes
:?????????

_user_specified_namex/10:MI
'
_output_shapes
:?????????

_user_specified_namex/11:MI
'
_output_shapes
:?????????

_user_specified_namex/12:MI
'
_output_shapes
:?????????

_user_specified_namex/13:MI
'
_output_shapes
:?????????

_user_specified_namex/14:MI
'
_output_shapes
:?????????

_user_specified_namex/15:MI
'
_output_shapes
:?????????

_user_specified_namex/16:MI
'
_output_shapes
:?????????

_user_specified_namex/17:MI
'
_output_shapes
:?????????

_user_specified_namex/18:MI
'
_output_shapes
:?????????

_user_specified_namex/19:MI
'
_output_shapes
:?????????

_user_specified_namex/20:MI
'
_output_shapes
:?????????

_user_specified_namex/21:MI
'
_output_shapes
:?????????

_user_specified_namex/22:MI
'
_output_shapes
:?????????

_user_specified_namex/23:MI
'
_output_shapes
:?????????

_user_specified_namex/24:MI
'
_output_shapes
:?????????

_user_specified_namex/25:MI
'
_output_shapes
:?????????

_user_specified_namex/26:MI
'
_output_shapes
:?????????

_user_specified_namex/27:MI
'
_output_shapes
:?????????

_user_specified_namex/28:MI
'
_output_shapes
:?????????

_user_specified_namex/29:MI
'
_output_shapes
:?????????

_user_specified_namex/30:MI
'
_output_shapes
:?????????

_user_specified_namex/31:M I
'
_output_shapes
:?????????

_user_specified_namex/32:M!I
'
_output_shapes
:?????????

_user_specified_namex/33:M"I
'
_output_shapes
:?????????

_user_specified_namex/34:M#I
'
_output_shapes
:?????????

_user_specified_namex/35:M$I
'
_output_shapes
:?????????

_user_specified_namex/36:M%I
'
_output_shapes
:?????????

_user_specified_namex/37:M&I
'
_output_shapes
:?????????

_user_specified_namex/38:M'I
'
_output_shapes
:?????????

_user_specified_namex/39:M(I
'
_output_shapes
:?????????

_user_specified_namex/40:M)I
'
_output_shapes
:?????????

_user_specified_namex/41:M*I
'
_output_shapes
:?????????

_user_specified_namex/42:M+I
'
_output_shapes
:?????????

_user_specified_namex/43:M,I
'
_output_shapes
:?????????

_user_specified_namex/44:M-I
'
_output_shapes
:?????????

_user_specified_namex/45:M.I
'
_output_shapes
:?????????

_user_specified_namex/46:M/I
'
_output_shapes
:?????????

_user_specified_namex/47:M0I
'
_output_shapes
:?????????

_user_specified_namex/48:M1I
'
_output_shapes
:?????????

_user_specified_namex/49
??
?	
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_423714
x_0
x_1
x_2
x_3
x_4
x_5
x_6
x_7
x_8
x_9
x_10
x_11
x_12
x_13
x_14
x_15
x_16
x_17
x_18
x_19
x_20
x_21
x_22
x_23
x_24
x_25
x_26
x_27
x_28
x_29
x_30
x_31
x_32
x_33
x_34
x_35
x_36
x_37
x_38
x_39
x_40
x_41
x_42
x_43
x_44
x_45
x_46
x_47
x_48
x_499
5sequential_16_dense_34_matmul_readvariableop_resource:
6sequential_16_dense_34_biasadd_readvariableop_resource9
5sequential_16_dense_35_matmul_readvariableop_resource:
6sequential_16_dense_35_biasadd_readvariableop_resource
readvariableop_resource
readvariableop_1_resource9
5sequential_17_dense_36_matmul_readvariableop_resource:
6sequential_17_dense_36_biasadd_readvariableop_resource9
5sequential_17_dense_37_matmul_readvariableop_resource:
6sequential_17_dense_37_biasadd_readvariableop_resource
identity

identity_1

identity_2

identity_3

identity_4??
,sequential_16/dense_34/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_16/dense_34/MatMul/ReadVariableOp?
sequential_16/dense_34/MatMulMatMulx_04sequential_16/dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/MatMul?
-sequential_16/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_16/dense_34/BiasAdd/ReadVariableOp?
sequential_16/dense_34/BiasAddBiasAdd'sequential_16/dense_34/MatMul:product:05sequential_16/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_16/dense_34/BiasAdd?
sequential_16/dense_34/SeluSelu'sequential_16/dense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/Selu?
,sequential_16/dense_35/MatMul/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_16/dense_35/MatMul/ReadVariableOp?
sequential_16/dense_35/MatMulMatMul)sequential_16/dense_34/Selu:activations:04sequential_16/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/MatMul?
-sequential_16/dense_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_16/dense_35/BiasAdd/ReadVariableOp?
sequential_16/dense_35/BiasAddBiasAdd'sequential_16/dense_35/MatMul:product:05sequential_16/dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_16/dense_35/BiasAdd?
sequential_16/dense_35/SeluSelu'sequential_16/dense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/Selup
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp?
mulMulReadVariableOp:value:0)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
mulw
SquareSquare)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
Squarev
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1m
mul_1MulReadVariableOp_1:value:0
Square:y:0*
T0*'
_output_shapes
:?????????2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add?
,sequential_17/dense_36/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_17/dense_36/MatMul/ReadVariableOp?
sequential_17/dense_36/MatMulMatMuladd:z:04sequential_17/dense_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/MatMul?
-sequential_17/dense_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02/
-sequential_17/dense_36/BiasAdd/ReadVariableOp?
sequential_17/dense_36/BiasAddBiasAdd'sequential_17/dense_36/MatMul:product:05sequential_17/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2 
sequential_17/dense_36/BiasAdd?
sequential_17/dense_36/SeluSelu'sequential_17/dense_36/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/Selu?
,sequential_17/dense_37/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02.
,sequential_17/dense_37/MatMul/ReadVariableOp?
sequential_17/dense_37/MatMulMatMul)sequential_17/dense_36/Selu:activations:04sequential_17/dense_37/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/MatMul?
-sequential_17/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_17/dense_37/BiasAdd/ReadVariableOp?
sequential_17/dense_37/BiasAddBiasAdd'sequential_17/dense_37/MatMul:product:05sequential_17/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_17/dense_37/BiasAdd?
sequential_17/dense_37/SeluSelu'sequential_17/dense_37/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/Selu?
.sequential_16/dense_34/MatMul_1/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_34/MatMul_1/ReadVariableOp?
sequential_16/dense_34/MatMul_1MatMulx_16sequential_16/dense_34/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_16/dense_34/MatMul_1?
/sequential_16/dense_34/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_16/dense_34/BiasAdd_1/ReadVariableOp?
 sequential_16/dense_34/BiasAdd_1BiasAdd)sequential_16/dense_34/MatMul_1:product:07sequential_16/dense_34/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_16/dense_34/BiasAdd_1?
sequential_16/dense_34/Selu_1Selu)sequential_16/dense_34/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/Selu_1?
.sequential_16/dense_35/MatMul_1/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_35/MatMul_1/ReadVariableOp?
sequential_16/dense_35/MatMul_1MatMul+sequential_16/dense_34/Selu_1:activations:06sequential_16/dense_35/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_16/dense_35/MatMul_1?
/sequential_16/dense_35/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_16/dense_35/BiasAdd_1/ReadVariableOp?
 sequential_16/dense_35/BiasAdd_1BiasAdd)sequential_16/dense_35/MatMul_1:product:07sequential_16/dense_35/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_16/dense_35/BiasAdd_1?
sequential_16/dense_35/Selu_1Selu)sequential_16/dense_35/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/Selu_1t
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_2?
mul_2MulReadVariableOp_2:value:0)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2
mul_2{
Square_1Square)sequential_16/dense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2

Square_1v
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_3o
mul_3MulReadVariableOp_3:value:0Square_1:y:0*
T0*'
_output_shapes
:?????????2
mul_3_
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*'
_output_shapes
:?????????2
add_1{
subSub+sequential_16/dense_35/Selu_1:activations:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
subY
Square_2Squaresub:z:0*
T0*'
_output_shapes
:?????????2

Square_2_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstS
MeanMeanSquare_2:y:0Const:output:0*
T0*
_output_shapes
: 2
Mean[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
	truediv/ya
truedivRealDivMean:output:0truediv/y:output:0*
T0*
_output_shapes
: 2	
truediv?
.sequential_17/dense_36/MatMul_1/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_36/MatMul_1/ReadVariableOp?
sequential_17/dense_36/MatMul_1MatMul	add_1:z:06sequential_17/dense_36/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_17/dense_36/MatMul_1?
/sequential_17/dense_36/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_17/dense_36/BiasAdd_1/ReadVariableOp?
 sequential_17/dense_36/BiasAdd_1BiasAdd)sequential_17/dense_36/MatMul_1:product:07sequential_17/dense_36/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_17/dense_36/BiasAdd_1?
sequential_17/dense_36/Selu_1Selu)sequential_17/dense_36/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/Selu_1?
.sequential_17/dense_37/MatMul_1/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_37/MatMul_1/ReadVariableOp?
sequential_17/dense_37/MatMul_1MatMul+sequential_17/dense_36/Selu_1:activations:06sequential_17/dense_37/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_17/dense_37/MatMul_1?
/sequential_17/dense_37/BiasAdd_1/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_17/dense_37/BiasAdd_1/ReadVariableOp?
 sequential_17/dense_37/BiasAdd_1BiasAdd)sequential_17/dense_37/MatMul_1:product:07sequential_17/dense_37/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_17/dense_37/BiasAdd_1?
sequential_17/dense_37/Selu_1Selu)sequential_17/dense_37/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/Selu_1y
sub_1Subx_1+sequential_17/dense_37/Selu_1:activations:0*
T0*'
_output_shapes
:?????????2
sub_1[
Square_3Square	sub_1:z:0*
T0*'
_output_shapes
:?????????2

Square_3c
Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_1Y
Mean_1MeanSquare_3:y:0Const_1:output:0*
T0*
_output_shapes
: 2
Mean_1_
truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_1/yi
	truediv_1RealDivMean_1:output:0truediv_1/y:output:0*
T0*
_output_shapes
: 2
	truediv_1?
.sequential_16/dense_34/MatMul_2/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_34/MatMul_2/ReadVariableOp?
sequential_16/dense_34/MatMul_2MatMulx_26sequential_16/dense_34/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_16/dense_34/MatMul_2?
/sequential_16/dense_34/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_16/dense_34/BiasAdd_2/ReadVariableOp?
 sequential_16/dense_34/BiasAdd_2BiasAdd)sequential_16/dense_34/MatMul_2:product:07sequential_16/dense_34/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_16/dense_34/BiasAdd_2?
sequential_16/dense_34/Selu_2Selu)sequential_16/dense_34/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????d2
sequential_16/dense_34/Selu_2?
.sequential_16/dense_35/MatMul_2/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_16/dense_35/MatMul_2/ReadVariableOp?
sequential_16/dense_35/MatMul_2MatMul+sequential_16/dense_34/Selu_2:activations:06sequential_16/dense_35/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_16/dense_35/MatMul_2?
/sequential_16/dense_35/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_16/dense_35/BiasAdd_2/ReadVariableOp?
 sequential_16/dense_35/BiasAdd_2BiasAdd)sequential_16/dense_35/MatMul_2:product:07sequential_16/dense_35/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_16/dense_35/BiasAdd_2?
sequential_16/dense_35/Selu_2Selu)sequential_16/dense_35/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????2
sequential_16/dense_35/Selu_2t
ReadVariableOp_4ReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOp_4l
mul_4MulReadVariableOp_4:value:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_4[
Square_4Square	add_1:z:0*
T0*'
_output_shapes
:?????????2

Square_4v
ReadVariableOp_5ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_5o
mul_5MulReadVariableOp_5:value:0Square_4:y:0*
T0*'
_output_shapes
:?????????2
mul_5_
add_2AddV2	mul_4:z:0	mul_5:z:0*
T0*'
_output_shapes
:?????????2
add_2
sub_2Sub+sequential_16/dense_35/Selu_2:activations:0	add_2:z:0*
T0*'
_output_shapes
:?????????2
sub_2[
Square_5Square	sub_2:z:0*
T0*'
_output_shapes
:?????????2

Square_5c
Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_2Y
Mean_2MeanSquare_5:y:0Const_2:output:0*
T0*
_output_shapes
: 2
Mean_2_
truediv_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_2/yi
	truediv_2RealDivMean_2:output:0truediv_2/y:output:0*
T0*
_output_shapes
: 2
	truediv_2?
.sequential_17/dense_36/MatMul_2/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_36/MatMul_2/ReadVariableOp?
sequential_17/dense_36/MatMul_2MatMul	add_2:z:06sequential_17/dense_36/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2!
sequential_17/dense_36/MatMul_2?
/sequential_17/dense_36/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/sequential_17/dense_36/BiasAdd_2/ReadVariableOp?
 sequential_17/dense_36/BiasAdd_2BiasAdd)sequential_17/dense_36/MatMul_2:product:07sequential_17/dense_36/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2"
 sequential_17/dense_36/BiasAdd_2?
sequential_17/dense_36/Selu_2Selu)sequential_17/dense_36/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????d2
sequential_17/dense_36/Selu_2?
.sequential_17/dense_37/MatMul_2/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.sequential_17/dense_37/MatMul_2/ReadVariableOp?
sequential_17/dense_37/MatMul_2MatMul+sequential_17/dense_36/Selu_2:activations:06sequential_17/dense_37/MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2!
sequential_17/dense_37/MatMul_2?
/sequential_17/dense_37/BiasAdd_2/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_17/dense_37/BiasAdd_2/ReadVariableOp?
 sequential_17/dense_37/BiasAdd_2BiasAdd)sequential_17/dense_37/MatMul_2:product:07sequential_17/dense_37/BiasAdd_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 sequential_17/dense_37/BiasAdd_2?
sequential_17/dense_37/Selu_2Selu)sequential_17/dense_37/BiasAdd_2:output:0*
T0*'
_output_shapes
:?????????2
sequential_17/dense_37/Selu_2y
sub_3Subx_2+sequential_17/dense_37/Selu_2:activations:0*
T0*'
_output_shapes
:?????????2
sub_3[
Square_6Square	sub_3:z:0*
T0*'
_output_shapes
:?????????2

Square_6c
Const_3Const*
_output_shapes
:*
dtype0*
valueB"       2	
Const_3Y
Mean_3MeanSquare_6:y:0Const_3:output:0*
T0*
_output_shapes
: 2
Mean_3_
truediv_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
truediv_3/yi
	truediv_3RealDivMean_3:output:0truediv_3/y:output:0*
T0*
_output_shapes
: 2
	truediv_3?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_16_dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_16_dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_36_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_dense_36_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5sequential_17_dense_37_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOp6sequential_17_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1}
IdentityIdentity)sequential_17/dense_37/Selu:activations:0*
T0*'
_output_shapes
:?????????2

IdentityR

Identity_1Identitytruediv:z:0*
T0*
_output_shapes
: 2

Identity_1T

Identity_2Identitytruediv_1:z:0*
T0*
_output_shapes
: 2

Identity_2T

Identity_3Identitytruediv_2:z:0*
T0*
_output_shapes
: 2

Identity_3T

Identity_4Identitytruediv_3:z:0*
T0*
_output_shapes
: 2

Identity_4"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:::::::::::L H
'
_output_shapes
:?????????

_user_specified_namex/0:LH
'
_output_shapes
:?????????

_user_specified_namex/1:LH
'
_output_shapes
:?????????

_user_specified_namex/2:LH
'
_output_shapes
:?????????

_user_specified_namex/3:LH
'
_output_shapes
:?????????

_user_specified_namex/4:LH
'
_output_shapes
:?????????

_user_specified_namex/5:LH
'
_output_shapes
:?????????

_user_specified_namex/6:LH
'
_output_shapes
:?????????

_user_specified_namex/7:LH
'
_output_shapes
:?????????

_user_specified_namex/8:L	H
'
_output_shapes
:?????????

_user_specified_namex/9:M
I
'
_output_shapes
:?????????

_user_specified_namex/10:MI
'
_output_shapes
:?????????

_user_specified_namex/11:MI
'
_output_shapes
:?????????

_user_specified_namex/12:MI
'
_output_shapes
:?????????

_user_specified_namex/13:MI
'
_output_shapes
:?????????

_user_specified_namex/14:MI
'
_output_shapes
:?????????

_user_specified_namex/15:MI
'
_output_shapes
:?????????

_user_specified_namex/16:MI
'
_output_shapes
:?????????

_user_specified_namex/17:MI
'
_output_shapes
:?????????

_user_specified_namex/18:MI
'
_output_shapes
:?????????

_user_specified_namex/19:MI
'
_output_shapes
:?????????

_user_specified_namex/20:MI
'
_output_shapes
:?????????

_user_specified_namex/21:MI
'
_output_shapes
:?????????

_user_specified_namex/22:MI
'
_output_shapes
:?????????

_user_specified_namex/23:MI
'
_output_shapes
:?????????

_user_specified_namex/24:MI
'
_output_shapes
:?????????

_user_specified_namex/25:MI
'
_output_shapes
:?????????

_user_specified_namex/26:MI
'
_output_shapes
:?????????

_user_specified_namex/27:MI
'
_output_shapes
:?????????

_user_specified_namex/28:MI
'
_output_shapes
:?????????

_user_specified_namex/29:MI
'
_output_shapes
:?????????

_user_specified_namex/30:MI
'
_output_shapes
:?????????

_user_specified_namex/31:M I
'
_output_shapes
:?????????

_user_specified_namex/32:M!I
'
_output_shapes
:?????????

_user_specified_namex/33:M"I
'
_output_shapes
:?????????

_user_specified_namex/34:M#I
'
_output_shapes
:?????????

_user_specified_namex/35:M$I
'
_output_shapes
:?????????

_user_specified_namex/36:M%I
'
_output_shapes
:?????????

_user_specified_namex/37:M&I
'
_output_shapes
:?????????

_user_specified_namex/38:M'I
'
_output_shapes
:?????????

_user_specified_namex/39:M(I
'
_output_shapes
:?????????

_user_specified_namex/40:M)I
'
_output_shapes
:?????????

_user_specified_namex/41:M*I
'
_output_shapes
:?????????

_user_specified_namex/42:M+I
'
_output_shapes
:?????????

_user_specified_namex/43:M,I
'
_output_shapes
:?????????

_user_specified_namex/44:M-I
'
_output_shapes
:?????????

_user_specified_namex/45:M.I
'
_output_shapes
:?????????

_user_specified_namex/46:M/I
'
_output_shapes
:?????????

_user_specified_namex/47:M0I
'
_output_shapes
:?????????

_user_specified_namex/48:M1I
'
_output_shapes
:?????????

_user_specified_namex/49
?]
?
I__inference_sequential_17_layer_call_and_return_conditional_losses_421934

inputs
dense_36_421863
dense_36_421865
dense_37_421868
dense_37_421870
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_421863dense_36_421865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_36_layer_call_and_return_conditional_losses_4215622"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_421868dense_37_421870*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_37_layer_call_and_return_conditional_losses_4216192"
 dense_37/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421863*
_output_shapes

:d*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421863*
_output_shapes

:d*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
dense_36/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_36/bias/Regularizer/Const?
,dense_36/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_421865*
_output_shapes
:d*
dtype02.
,dense_36/bias/Regularizer/Abs/ReadVariableOp?
dense_36/bias/Regularizer/AbsAbs4dense_36/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_36/bias/Regularizer/Abs?
!dense_36/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_1?
dense_36/bias/Regularizer/SumSum!dense_36/bias/Regularizer/Abs:y:0*dense_36/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/Sum?
dense_36/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_36/bias/Regularizer/mul/x?
dense_36/bias/Regularizer/mulMul(dense_36/bias/Regularizer/mul/x:output:0&dense_36/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/mul?
dense_36/bias/Regularizer/addAddV2(dense_36/bias/Regularizer/Const:output:0!dense_36/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_36/bias/Regularizer/add?
/dense_36/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_421865*
_output_shapes
:d*
dtype021
/dense_36/bias/Regularizer/Square/ReadVariableOp?
 dense_36/bias/Regularizer/SquareSquare7dense_36/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_36/bias/Regularizer/Square?
!dense_36/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_36/bias/Regularizer/Const_2?
dense_36/bias/Regularizer/Sum_1Sum$dense_36/bias/Regularizer/Square:y:0*dense_36/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/Sum_1?
!dense_36/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_36/bias/Regularizer/mul_1/x?
dense_36/bias/Regularizer/mul_1Mul*dense_36/bias/Regularizer/mul_1/x:output:0(dense_36/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/mul_1?
dense_36/bias/Regularizer/add_1AddV2!dense_36/bias/Regularizer/add:z:0#dense_36/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_36/bias/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421868*
_output_shapes

:d*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421868*
_output_shapes

:d*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
dense_37/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_37/bias/Regularizer/Const?
,dense_37/bias/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_421870*
_output_shapes
:*
dtype02.
,dense_37/bias/Regularizer/Abs/ReadVariableOp?
dense_37/bias/Regularizer/AbsAbs4dense_37/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_37/bias/Regularizer/Abs?
!dense_37/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_1?
dense_37/bias/Regularizer/SumSum!dense_37/bias/Regularizer/Abs:y:0*dense_37/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/Sum?
dense_37/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_37/bias/Regularizer/mul/x?
dense_37/bias/Regularizer/mulMul(dense_37/bias/Regularizer/mul/x:output:0&dense_37/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/mul?
dense_37/bias/Regularizer/addAddV2(dense_37/bias/Regularizer/Const:output:0!dense_37/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_37/bias/Regularizer/add?
/dense_37/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_421870*
_output_shapes
:*
dtype021
/dense_37/bias/Regularizer/Square/ReadVariableOp?
 dense_37/bias/Regularizer/SquareSquare7dense_37/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_37/bias/Regularizer/Square?
!dense_37/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_37/bias/Regularizer/Const_2?
dense_37/bias/Regularizer/Sum_1Sum$dense_37/bias/Regularizer/Square:y:0*dense_37/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/Sum_1?
!dense_37/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_37/bias/Regularizer/mul_1/x?
dense_37/bias/Regularizer/mul_1Mul*dense_37/bias/Regularizer/mul_1/x:output:0(dense_37/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/mul_1?
dense_37/bias/Regularizer/add_1AddV2!dense_37/bias/Regularizer/add:z:0#dense_37/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_37/bias/Regularizer/add_1?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
__inference_loss_fn_3_4245949
5dense_35_bias_regularizer_abs_readvariableop_resource
identity??
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOp5dense_35_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp5dense_35_bias_regularizer_abs_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1f
IdentityIdentity#dense_35/bias/Regularizer/add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
?b
?
I__inference_sequential_16_layer_call_and_return_conditional_losses_424008

inputs+
'dense_34_matmul_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource+
'dense_35_matmul_readvariableop_resource,
(dense_35_biasadd_readvariableop_resource
identity??
dense_34/MatMul/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_34/MatMul/ReadVariableOp?
dense_34/MatMulMatMulinputs&dense_34/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_34/MatMul?
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02!
dense_34/BiasAdd/ReadVariableOp?
dense_34/BiasAddBiasAdddense_34/MatMul:product:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d2
dense_34/BiasAdds
dense_34/SeluSeludense_34/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d2
dense_34/Selu?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02 
dense_35/MatMul/ReadVariableOp?
dense_35/MatMulMatMuldense_34/Selu:activations:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/MatMul?
dense_35/BiasAdd/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_35/BiasAdd/ReadVariableOp?
dense_35/BiasAddBiasAdddense_35/MatMul:product:0'dense_35/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_35/BiasAdds
dense_35/SeluSeludense_35/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_35/Selu?
!dense_34/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_34/kernel/Regularizer/Const?
.dense_34/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_34/kernel/Regularizer/Abs/ReadVariableOp?
dense_34/kernel/Regularizer/AbsAbs6dense_34/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_34/kernel/Regularizer/Abs?
#dense_34/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_1?
dense_34/kernel/Regularizer/SumSum#dense_34/kernel/Regularizer/Abs:y:0,dense_34/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/Sum?
!dense_34/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/kernel/Regularizer/mul/x?
dense_34/kernel/Regularizer/mulMul*dense_34/kernel/Regularizer/mul/x:output:0(dense_34/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/mul?
dense_34/kernel/Regularizer/addAddV2*dense_34/kernel/Regularizer/Const:output:0#dense_34/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_34/kernel/Regularizer/add?
1dense_34/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_34_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_34/kernel/Regularizer/Square/ReadVariableOp?
"dense_34/kernel/Regularizer/SquareSquare9dense_34/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_34/kernel/Regularizer/Square?
#dense_34/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_34/kernel/Regularizer/Const_2?
!dense_34/kernel/Regularizer/Sum_1Sum&dense_34/kernel/Regularizer/Square:y:0,dense_34/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/Sum_1?
#dense_34/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_34/kernel/Regularizer/mul_1/x?
!dense_34/kernel/Regularizer/mul_1Mul,dense_34/kernel/Regularizer/mul_1/x:output:0*dense_34/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/mul_1?
!dense_34/kernel/Regularizer/add_1AddV2#dense_34/kernel/Regularizer/add:z:0%dense_34/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_34/kernel/Regularizer/add_1?
dense_34/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_34/bias/Regularizer/Const?
,dense_34/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02.
,dense_34/bias/Regularizer/Abs/ReadVariableOp?
dense_34/bias/Regularizer/AbsAbs4dense_34/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:d2
dense_34/bias/Regularizer/Abs?
!dense_34/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_1?
dense_34/bias/Regularizer/SumSum!dense_34/bias/Regularizer/Abs:y:0*dense_34/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/Sum?
dense_34/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_34/bias/Regularizer/mul/x?
dense_34/bias/Regularizer/mulMul(dense_34/bias/Regularizer/mul/x:output:0&dense_34/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/mul?
dense_34/bias/Regularizer/addAddV2(dense_34/bias/Regularizer/Const:output:0!dense_34/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_34/bias/Regularizer/add?
/dense_34/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype021
/dense_34/bias/Regularizer/Square/ReadVariableOp?
 dense_34/bias/Regularizer/SquareSquare7dense_34/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:d2"
 dense_34/bias/Regularizer/Square?
!dense_34/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_34/bias/Regularizer/Const_2?
dense_34/bias/Regularizer/Sum_1Sum$dense_34/bias/Regularizer/Square:y:0*dense_34/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/Sum_1?
!dense_34/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_34/bias/Regularizer/mul_1/x?
dense_34/bias/Regularizer/mul_1Mul*dense_34/bias/Regularizer/mul_1/x:output:0(dense_34/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/mul_1?
dense_34/bias/Regularizer/add_1AddV2!dense_34/bias/Regularizer/add:z:0#dense_34/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_34/bias/Regularizer/add_1?
!dense_35/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_35/kernel/Regularizer/Const?
.dense_35/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype020
.dense_35/kernel/Regularizer/Abs/ReadVariableOp?
dense_35/kernel/Regularizer/AbsAbs6dense_35/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:d2!
dense_35/kernel/Regularizer/Abs?
#dense_35/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_1?
dense_35/kernel/Regularizer/SumSum#dense_35/kernel/Regularizer/Abs:y:0,dense_35/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/Sum?
!dense_35/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/kernel/Regularizer/mul/x?
dense_35/kernel/Regularizer/mulMul*dense_35/kernel/Regularizer/mul/x:output:0(dense_35/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/mul?
dense_35/kernel/Regularizer/addAddV2*dense_35/kernel/Regularizer/Const:output:0#dense_35/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_35/kernel/Regularizer/add?
1dense_35/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d*
dtype023
1dense_35/kernel/Regularizer/Square/ReadVariableOp?
"dense_35/kernel/Regularizer/SquareSquare9dense_35/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:d2$
"dense_35/kernel/Regularizer/Square?
#dense_35/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_35/kernel/Regularizer/Const_2?
!dense_35/kernel/Regularizer/Sum_1Sum&dense_35/kernel/Regularizer/Square:y:0,dense_35/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/Sum_1?
#dense_35/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2%
#dense_35/kernel/Regularizer/mul_1/x?
!dense_35/kernel/Regularizer/mul_1Mul,dense_35/kernel/Regularizer/mul_1/x:output:0*dense_35/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/mul_1?
!dense_35/kernel/Regularizer/add_1AddV2#dense_35/kernel/Regularizer/add:z:0%dense_35/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_35/kernel/Regularizer/add_1?
dense_35/bias/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_35/bias/Regularizer/Const?
,dense_35/bias/Regularizer/Abs/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,dense_35/bias/Regularizer/Abs/ReadVariableOp?
dense_35/bias/Regularizer/AbsAbs4dense_35/bias/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:2
dense_35/bias/Regularizer/Abs?
!dense_35/bias/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_1?
dense_35/bias/Regularizer/SumSum!dense_35/bias/Regularizer/Abs:y:0*dense_35/bias/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/Sum?
dense_35/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2!
dense_35/bias/Regularizer/mul/x?
dense_35/bias/Regularizer/mulMul(dense_35/bias/Regularizer/mul/x:output:0&dense_35/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/mul?
dense_35/bias/Regularizer/addAddV2(dense_35/bias/Regularizer/Const:output:0!dense_35/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
dense_35/bias/Regularizer/add?
/dense_35/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/dense_35/bias/Regularizer/Square/ReadVariableOp?
 dense_35/bias/Regularizer/SquareSquare7dense_35/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2"
 dense_35/bias/Regularizer/Square?
!dense_35/bias/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2#
!dense_35/bias/Regularizer/Const_2?
dense_35/bias/Regularizer/Sum_1Sum$dense_35/bias/Regularizer/Square:y:0*dense_35/bias/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/Sum_1?
!dense_35/bias/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *???.2#
!dense_35/bias/Regularizer/mul_1/x?
dense_35/bias/Regularizer/mul_1Mul*dense_35/bias/Regularizer/mul_1/x:output:0(dense_35/bias/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/mul_1?
dense_35/bias/Regularizer/add_1AddV2!dense_35/bias/Regularizer/add:z:0#dense_35/bias/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense_35/bias/Regularizer/add_1o
IdentityIdentitydense_35/Selu:activations:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::::O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
=
input_101
serving_default_input_10:0?????????
=
input_111
serving_default_input_11:0?????????
=
input_121
serving_default_input_12:0?????????
=
input_131
serving_default_input_13:0?????????
=
input_141
serving_default_input_14:0?????????
=
input_151
serving_default_input_15:0?????????
=
input_161
serving_default_input_16:0?????????
=
input_171
serving_default_input_17:0?????????
=
input_181
serving_default_input_18:0?????????
=
input_191
serving_default_input_19:0?????????
;
input_20
serving_default_input_2:0?????????
=
input_201
serving_default_input_20:0?????????
=
input_211
serving_default_input_21:0?????????
=
input_221
serving_default_input_22:0?????????
=
input_231
serving_default_input_23:0?????????
=
input_241
serving_default_input_24:0?????????
=
input_251
serving_default_input_25:0?????????
=
input_261
serving_default_input_26:0?????????
=
input_271
serving_default_input_27:0?????????
=
input_281
serving_default_input_28:0?????????
=
input_291
serving_default_input_29:0?????????
;
input_30
serving_default_input_3:0?????????
=
input_301
serving_default_input_30:0?????????
=
input_311
serving_default_input_31:0?????????
=
input_321
serving_default_input_32:0?????????
=
input_331
serving_default_input_33:0?????????
=
input_341
serving_default_input_34:0?????????
=
input_351
serving_default_input_35:0?????????
=
input_361
serving_default_input_36:0?????????
=
input_371
serving_default_input_37:0?????????
=
input_381
serving_default_input_38:0?????????
=
input_391
serving_default_input_39:0?????????
;
input_40
serving_default_input_4:0?????????
=
input_401
serving_default_input_40:0?????????
=
input_411
serving_default_input_41:0?????????
=
input_421
serving_default_input_42:0?????????
=
input_431
serving_default_input_43:0?????????
=
input_441
serving_default_input_44:0?????????
=
input_451
serving_default_input_45:0?????????
=
input_461
serving_default_input_46:0?????????
=
input_471
serving_default_input_47:0?????????
=
input_481
serving_default_input_48:0?????????
=
input_491
serving_default_input_49:0?????????
;
input_50
serving_default_input_5:0?????????
=
input_501
serving_default_input_50:0?????????
;
input_60
serving_default_input_6:0?????????
;
input_70
serving_default_input_7:0?????????
;
input_80
serving_default_input_8:0?????????
;
input_90
serving_default_input_9:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
c1
c2
encoder
decoder
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
u__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses"?
_tf_keras_model?{"class_name": "Conjugacy", "name": "conjugacy_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Conjugacy"}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
: 2Variable
: 2Variable
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
_build_input_shape
regularization_losses
	variables
trainable_variables
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_34_input"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": [{"class_name": "__tuple__", "items": []}, {"class_name": "__tuple__", "items": []}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_34_input"}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
regularization_losses
	variables
trainable_variables
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_17", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_36_input"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?
iter

beta_1

beta_2
	decay
learning_ratemambmcmdme mf!mg"mh#mi$mjvkvlvmvnvo vp!vq"vr#vs$vt"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
 3
!4
"5
#6
$7
8
9"
trackable_list_wrapper
f
0
1
2
 3
!4
"5
#6
$7
8
9"
trackable_list_wrapper
?
%non_trainable_variables
&metrics
'layer_metrics
regularization_losses
(layer_regularization_losses

)layers
	variables
trainable_variables
u__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
,
|serving_default"
signature_map
?	
*_inbound_nodes

kernel
bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?	
/_inbound_nodes

kernel
 bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_35", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_35", "trainable": true, "dtype": "float32", "units": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
?
4non_trainable_variables
5metrics
6layer_metrics
regularization_losses
7layer_regularization_losses

8layers
	variables
trainable_variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
?	
9_inbound_nodes

!kernel
"bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 100, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?	
>_inbound_nodes

#kernel
$bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 1, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.5, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 1.000000013351432e-10, "l2": 1.000000013351432e-10}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
<
!0
"1
#2
$3"
trackable_list_wrapper
?
Cnon_trainable_variables
Dmetrics
Elayer_metrics
regularization_losses
Flayer_regularization_losses

Glayers
	variables
trainable_variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
!:d2dense_34/kernel
:d2dense_34/bias
!:d2dense_35/kernel
:2dense_35/bias
!:d2dense_36/kernel
:d2dense_36/bias
!:d2dense_37/kernel
:2dense_37/bias
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Inon_trainable_variables
Jmetrics
Klayer_metrics
+regularization_losses
Llayer_regularization_losses

Mlayers
,	variables
-trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
Nnon_trainable_variables
Ometrics
Player_metrics
0regularization_losses
Qlayer_regularization_losses

Rlayers
1	variables
2trainable_variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
?
Snon_trainable_variables
Tmetrics
Ulayer_metrics
:regularization_losses
Vlayer_regularization_losses

Wlayers
;	variables
<trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
Xnon_trainable_variables
Ymetrics
Zlayer_metrics
?regularization_losses
[layer_regularization_losses

\layers
@	variables
Atrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	]total
	^count
_	variables
`	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
]0
^1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
: 2Adam/Variable/m
: 2Adam/Variable/m
&:$d2Adam/dense_34/kernel/m
 :d2Adam/dense_34/bias/m
&:$d2Adam/dense_35/kernel/m
 :2Adam/dense_35/bias/m
&:$d2Adam/dense_36/kernel/m
 :d2Adam/dense_36/bias/m
&:$d2Adam/dense_37/kernel/m
 :2Adam/dense_37/bias/m
: 2Adam/Variable/v
: 2Adam/Variable/v
&:$d2Adam/dense_34/kernel/v
 :d2Adam/dense_34/bias/v
&:$d2Adam/dense_35/kernel/v
 :2Adam/dense_35/bias/v
&:$d2Adam/dense_36/kernel/v
 :d2Adam/dense_36/bias/v
&:$d2Adam/dense_37/kernel/v
 :2Adam/dense_37/bias/v
?2?
,__inference_conjugacy_8_layer_call_fn_422854
,__inference_conjugacy_8_layer_call_fn_423870
,__inference_conjugacy_8_layer_call_fn_422932
,__inference_conjugacy_8_layer_call_fn_423792?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
!__inference__wrapped_model_421089?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *???
???
!?
input_1?????????
!?
input_2?????????
!?
input_3?????????
!?
input_4?????????
!?
input_5?????????
!?
input_6?????????
!?
input_7?????????
!?
input_8?????????
!?
input_9?????????
"?
input_10?????????
"?
input_11?????????
"?
input_12?????????
"?
input_13?????????
"?
input_14?????????
"?
input_15?????????
"?
input_16?????????
"?
input_17?????????
"?
input_18?????????
"?
input_19?????????
"?
input_20?????????
"?
input_21?????????
"?
input_22?????????
"?
input_23?????????
"?
input_24?????????
"?
input_25?????????
"?
input_26?????????
"?
input_27?????????
"?
input_28?????????
"?
input_29?????????
"?
input_30?????????
"?
input_31?????????
"?
input_32?????????
"?
input_33?????????
"?
input_34?????????
"?
input_35?????????
"?
input_36?????????
"?
input_37?????????
"?
input_38?????????
"?
input_39?????????
"?
input_40?????????
"?
input_41?????????
"?
input_42?????????
"?
input_43?????????
"?
input_44?????????
"?
input_45?????????
"?
input_46?????????
"?
input_47?????????
"?
input_48?????????
"?
input_49?????????
"?
input_50?????????
?2?
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_423714
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_422516
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_422257
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_423425?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_sequential_16_layer_call_fn_424099
.__inference_sequential_16_layer_call_fn_424112
.__inference_sequential_16_layer_call_fn_421517
.__inference_sequential_16_layer_call_fn_421430?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_16_layer_call_and_return_conditional_losses_424086
I__inference_sequential_16_layer_call_and_return_conditional_losses_421268
I__inference_sequential_16_layer_call_and_return_conditional_losses_424008
I__inference_sequential_16_layer_call_and_return_conditional_losses_421342?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_17_layer_call_fn_421945
.__inference_sequential_17_layer_call_fn_424341
.__inference_sequential_17_layer_call_fn_421858
.__inference_sequential_17_layer_call_fn_424354?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_17_layer_call_and_return_conditional_losses_421696
I__inference_sequential_17_layer_call_and_return_conditional_losses_421770
I__inference_sequential_17_layer_call_and_return_conditional_losses_424328
I__inference_sequential_17_layer_call_and_return_conditional_losses_424250?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_423136input_1input_10input_11input_12input_13input_14input_15input_16input_17input_18input_19input_2input_20input_21input_22input_23input_24input_25input_26input_27input_28input_29input_3input_30input_31input_32input_33input_34input_35input_36input_37input_38input_39input_4input_40input_41input_42input_43input_44input_45input_46input_47input_48input_49input_5input_50input_6input_7input_8input_9
?2?
)__inference_dense_34_layer_call_fn_424434?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_34_layer_call_and_return_conditional_losses_424425?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_35_layer_call_fn_424514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_35_layer_call_and_return_conditional_losses_424505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_424534?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_424554?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_424574?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_424594?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
)__inference_dense_36_layer_call_fn_424674?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_36_layer_call_and_return_conditional_losses_424665?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_37_layer_call_fn_424754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_37_layer_call_and_return_conditional_losses_424745?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_4_424774?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_424794?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_6_424814?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_7_424834?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? ?
!__inference__wrapped_model_421089?
 !"#$???
???
???
!?
input_1?????????
!?
input_2?????????
!?
input_3?????????
!?
input_4?????????
!?
input_5?????????
!?
input_6?????????
!?
input_7?????????
!?
input_8?????????
!?
input_9?????????
"?
input_10?????????
"?
input_11?????????
"?
input_12?????????
"?
input_13?????????
"?
input_14?????????
"?
input_15?????????
"?
input_16?????????
"?
input_17?????????
"?
input_18?????????
"?
input_19?????????
"?
input_20?????????
"?
input_21?????????
"?
input_22?????????
"?
input_23?????????
"?
input_24?????????
"?
input_25?????????
"?
input_26?????????
"?
input_27?????????
"?
input_28?????????
"?
input_29?????????
"?
input_30?????????
"?
input_31?????????
"?
input_32?????????
"?
input_33?????????
"?
input_34?????????
"?
input_35?????????
"?
input_36?????????
"?
input_37?????????
"?
input_38?????????
"?
input_39?????????
"?
input_40?????????
"?
input_41?????????
"?
input_42?????????
"?
input_43?????????
"?
input_44?????????
"?
input_45?????????
"?
input_46?????????
"?
input_47?????????
"?
input_48?????????
"?
input_49?????????
"?
input_50?????????
? "3?0
.
output_1"?
output_1??????????
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_422257?
 !"#$???
???
???
!?
input_1?????????
!?
input_2?????????
!?
input_3?????????
!?
input_4?????????
!?
input_5?????????
!?
input_6?????????
!?
input_7?????????
!?
input_8?????????
!?
input_9?????????
"?
input_10?????????
"?
input_11?????????
"?
input_12?????????
"?
input_13?????????
"?
input_14?????????
"?
input_15?????????
"?
input_16?????????
"?
input_17?????????
"?
input_18?????????
"?
input_19?????????
"?
input_20?????????
"?
input_21?????????
"?
input_22?????????
"?
input_23?????????
"?
input_24?????????
"?
input_25?????????
"?
input_26?????????
"?
input_27?????????
"?
input_28?????????
"?
input_29?????????
"?
input_30?????????
"?
input_31?????????
"?
input_32?????????
"?
input_33?????????
"?
input_34?????????
"?
input_35?????????
"?
input_36?????????
"?
input_37?????????
"?
input_38?????????
"?
input_39?????????
"?
input_40?????????
"?
input_41?????????
"?
input_42?????????
"?
input_43?????????
"?
input_44?????????
"?
input_45?????????
"?
input_46?????????
"?
input_47?????????
"?
input_48?????????
"?
input_49?????????
"?
input_50?????????
p
? "]?Z
?
0?????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_422516?
 !"#$???
???
???
!?
input_1?????????
!?
input_2?????????
!?
input_3?????????
!?
input_4?????????
!?
input_5?????????
!?
input_6?????????
!?
input_7?????????
!?
input_8?????????
!?
input_9?????????
"?
input_10?????????
"?
input_11?????????
"?
input_12?????????
"?
input_13?????????
"?
input_14?????????
"?
input_15?????????
"?
input_16?????????
"?
input_17?????????
"?
input_18?????????
"?
input_19?????????
"?
input_20?????????
"?
input_21?????????
"?
input_22?????????
"?
input_23?????????
"?
input_24?????????
"?
input_25?????????
"?
input_26?????????
"?
input_27?????????
"?
input_28?????????
"?
input_29?????????
"?
input_30?????????
"?
input_31?????????
"?
input_32?????????
"?
input_33?????????
"?
input_34?????????
"?
input_35?????????
"?
input_36?????????
"?
input_37?????????
"?
input_38?????????
"?
input_39?????????
"?
input_40?????????
"?
input_41?????????
"?
input_42?????????
"?
input_43?????????
"?
input_44?????????
"?
input_45?????????
"?
input_46?????????
"?
input_47?????????
"?
input_48?????????
"?
input_49?????????
"?
input_50?????????
p 
? "]?Z
?
0?????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_423425?
 !"#$???
???
???
?
x/0?????????
?
x/1?????????
?
x/2?????????
?
x/3?????????
?
x/4?????????
?
x/5?????????
?
x/6?????????
?
x/7?????????
?
x/8?????????
?
x/9?????????
?
x/10?????????
?
x/11?????????
?
x/12?????????
?
x/13?????????
?
x/14?????????
?
x/15?????????
?
x/16?????????
?
x/17?????????
?
x/18?????????
?
x/19?????????
?
x/20?????????
?
x/21?????????
?
x/22?????????
?
x/23?????????
?
x/24?????????
?
x/25?????????
?
x/26?????????
?
x/27?????????
?
x/28?????????
?
x/29?????????
?
x/30?????????
?
x/31?????????
?
x/32?????????
?
x/33?????????
?
x/34?????????
?
x/35?????????
?
x/36?????????
?
x/37?????????
?
x/38?????????
?
x/39?????????
?
x/40?????????
?
x/41?????????
?
x/42?????????
?
x/43?????????
?
x/44?????????
?
x/45?????????
?
x/46?????????
?
x/47?????????
?
x/48?????????
?
x/49?????????
p
? "]?Z
?
0?????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
G__inference_conjugacy_8_layer_call_and_return_conditional_losses_423714?
 !"#$???
???
???
?
x/0?????????
?
x/1?????????
?
x/2?????????
?
x/3?????????
?
x/4?????????
?
x/5?????????
?
x/6?????????
?
x/7?????????
?
x/8?????????
?
x/9?????????
?
x/10?????????
?
x/11?????????
?
x/12?????????
?
x/13?????????
?
x/14?????????
?
x/15?????????
?
x/16?????????
?
x/17?????????
?
x/18?????????
?
x/19?????????
?
x/20?????????
?
x/21?????????
?
x/22?????????
?
x/23?????????
?
x/24?????????
?
x/25?????????
?
x/26?????????
?
x/27?????????
?
x/28?????????
?
x/29?????????
?
x/30?????????
?
x/31?????????
?
x/32?????????
?
x/33?????????
?
x/34?????????
?
x/35?????????
?
x/36?????????
?
x/37?????????
?
x/38?????????
?
x/39?????????
?
x/40?????????
?
x/41?????????
?
x/42?????????
?
x/43?????????
?
x/44?????????
?
x/45?????????
?
x/46?????????
?
x/47?????????
?
x/48?????????
?
x/49?????????
p 
? "]?Z
?
0?????????
;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
,__inference_conjugacy_8_layer_call_fn_422854?
 !"#$???
???
???
!?
input_1?????????
!?
input_2?????????
!?
input_3?????????
!?
input_4?????????
!?
input_5?????????
!?
input_6?????????
!?
input_7?????????
!?
input_8?????????
!?
input_9?????????
"?
input_10?????????
"?
input_11?????????
"?
input_12?????????
"?
input_13?????????
"?
input_14?????????
"?
input_15?????????
"?
input_16?????????
"?
input_17?????????
"?
input_18?????????
"?
input_19?????????
"?
input_20?????????
"?
input_21?????????
"?
input_22?????????
"?
input_23?????????
"?
input_24?????????
"?
input_25?????????
"?
input_26?????????
"?
input_27?????????
"?
input_28?????????
"?
input_29?????????
"?
input_30?????????
"?
input_31?????????
"?
input_32?????????
"?
input_33?????????
"?
input_34?????????
"?
input_35?????????
"?
input_36?????????
"?
input_37?????????
"?
input_38?????????
"?
input_39?????????
"?
input_40?????????
"?
input_41?????????
"?
input_42?????????
"?
input_43?????????
"?
input_44?????????
"?
input_45?????????
"?
input_46?????????
"?
input_47?????????
"?
input_48?????????
"?
input_49?????????
"?
input_50?????????
p
? "???????????
,__inference_conjugacy_8_layer_call_fn_422932?
 !"#$???
???
???
!?
input_1?????????
!?
input_2?????????
!?
input_3?????????
!?
input_4?????????
!?
input_5?????????
!?
input_6?????????
!?
input_7?????????
!?
input_8?????????
!?
input_9?????????
"?
input_10?????????
"?
input_11?????????
"?
input_12?????????
"?
input_13?????????
"?
input_14?????????
"?
input_15?????????
"?
input_16?????????
"?
input_17?????????
"?
input_18?????????
"?
input_19?????????
"?
input_20?????????
"?
input_21?????????
"?
input_22?????????
"?
input_23?????????
"?
input_24?????????
"?
input_25?????????
"?
input_26?????????
"?
input_27?????????
"?
input_28?????????
"?
input_29?????????
"?
input_30?????????
"?
input_31?????????
"?
input_32?????????
"?
input_33?????????
"?
input_34?????????
"?
input_35?????????
"?
input_36?????????
"?
input_37?????????
"?
input_38?????????
"?
input_39?????????
"?
input_40?????????
"?
input_41?????????
"?
input_42?????????
"?
input_43?????????
"?
input_44?????????
"?
input_45?????????
"?
input_46?????????
"?
input_47?????????
"?
input_48?????????
"?
input_49?????????
"?
input_50?????????
p 
? "???????????
,__inference_conjugacy_8_layer_call_fn_423792?
 !"#$???
???
???
?
x/0?????????
?
x/1?????????
?
x/2?????????
?
x/3?????????
?
x/4?????????
?
x/5?????????
?
x/6?????????
?
x/7?????????
?
x/8?????????
?
x/9?????????
?
x/10?????????
?
x/11?????????
?
x/12?????????
?
x/13?????????
?
x/14?????????
?
x/15?????????
?
x/16?????????
?
x/17?????????
?
x/18?????????
?
x/19?????????
?
x/20?????????
?
x/21?????????
?
x/22?????????
?
x/23?????????
?
x/24?????????
?
x/25?????????
?
x/26?????????
?
x/27?????????
?
x/28?????????
?
x/29?????????
?
x/30?????????
?
x/31?????????
?
x/32?????????
?
x/33?????????
?
x/34?????????
?
x/35?????????
?
x/36?????????
?
x/37?????????
?
x/38?????????
?
x/39?????????
?
x/40?????????
?
x/41?????????
?
x/42?????????
?
x/43?????????
?
x/44?????????
?
x/45?????????
?
x/46?????????
?
x/47?????????
?
x/48?????????
?
x/49?????????
p
? "???????????
,__inference_conjugacy_8_layer_call_fn_423870?
 !"#$???
???
???
?
x/0?????????
?
x/1?????????
?
x/2?????????
?
x/3?????????
?
x/4?????????
?
x/5?????????
?
x/6?????????
?
x/7?????????
?
x/8?????????
?
x/9?????????
?
x/10?????????
?
x/11?????????
?
x/12?????????
?
x/13?????????
?
x/14?????????
?
x/15?????????
?
x/16?????????
?
x/17?????????
?
x/18?????????
?
x/19?????????
?
x/20?????????
?
x/21?????????
?
x/22?????????
?
x/23?????????
?
x/24?????????
?
x/25?????????
?
x/26?????????
?
x/27?????????
?
x/28?????????
?
x/29?????????
?
x/30?????????
?
x/31?????????
?
x/32?????????
?
x/33?????????
?
x/34?????????
?
x/35?????????
?
x/36?????????
?
x/37?????????
?
x/38?????????
?
x/39?????????
?
x/40?????????
?
x/41?????????
?
x/42?????????
?
x/43?????????
?
x/44?????????
?
x/45?????????
?
x/46?????????
?
x/47?????????
?
x/48?????????
?
x/49?????????
p 
? "???????????
D__inference_dense_34_layer_call_and_return_conditional_losses_424425\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????d
? |
)__inference_dense_34_layer_call_fn_424434O/?,
%?"
 ?
inputs?????????
? "??????????d?
D__inference_dense_35_layer_call_and_return_conditional_losses_424505\ /?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? |
)__inference_dense_35_layer_call_fn_424514O /?,
%?"
 ?
inputs?????????d
? "???????????
D__inference_dense_36_layer_call_and_return_conditional_losses_424665\!"/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????d
? |
)__inference_dense_36_layer_call_fn_424674O!"/?,
%?"
 ?
inputs?????????
? "??????????d?
D__inference_dense_37_layer_call_and_return_conditional_losses_424745\#$/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????
? |
)__inference_dense_37_layer_call_fn_424754O#$/?,
%?"
 ?
inputs?????????d
? "??????????;
__inference_loss_fn_0_424534?

? 
? "? ;
__inference_loss_fn_1_424554?

? 
? "? ;
__inference_loss_fn_2_424574?

? 
? "? ;
__inference_loss_fn_3_424594 ?

? 
? "? ;
__inference_loss_fn_4_424774!?

? 
? "? ;
__inference_loss_fn_5_424794"?

? 
? "? ;
__inference_loss_fn_6_424814#?

? 
? "? ;
__inference_loss_fn_7_424834$?

? 
? "? ?
I__inference_sequential_16_layer_call_and_return_conditional_losses_421268n ??<
5?2
(?%
dense_34_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_16_layer_call_and_return_conditional_losses_421342n ??<
5?2
(?%
dense_34_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_16_layer_call_and_return_conditional_losses_424008f 7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_16_layer_call_and_return_conditional_losses_424086f 7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_16_layer_call_fn_421430a ??<
5?2
(?%
dense_34_input?????????
p

 
? "???????????
.__inference_sequential_16_layer_call_fn_421517a ??<
5?2
(?%
dense_34_input?????????
p 

 
? "???????????
.__inference_sequential_16_layer_call_fn_424099Y 7?4
-?*
 ?
inputs?????????
p

 
? "???????????
.__inference_sequential_16_layer_call_fn_424112Y 7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
I__inference_sequential_17_layer_call_and_return_conditional_losses_421696n!"#$??<
5?2
(?%
dense_36_input?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_421770n!"#$??<
5?2
(?%
dense_36_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_424250f!"#$7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_17_layer_call_and_return_conditional_losses_424328f!"#$7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_17_layer_call_fn_421858a!"#$??<
5?2
(?%
dense_36_input?????????
p

 
? "???????????
.__inference_sequential_17_layer_call_fn_421945a!"#$??<
5?2
(?%
dense_36_input?????????
p 

 
? "???????????
.__inference_sequential_17_layer_call_fn_424341Y!"#$7?4
-?*
 ?
inputs?????????
p

 
? "???????????
.__inference_sequential_17_layer_call_fn_424354Y!"#$7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
$__inference_signature_wrapper_423136?
 !"#$???
? 
???
,
input_1!?
input_1?????????
.
input_10"?
input_10?????????
.
input_11"?
input_11?????????
.
input_12"?
input_12?????????
.
input_13"?
input_13?????????
.
input_14"?
input_14?????????
.
input_15"?
input_15?????????
.
input_16"?
input_16?????????
.
input_17"?
input_17?????????
.
input_18"?
input_18?????????
.
input_19"?
input_19?????????
,
input_2!?
input_2?????????
.
input_20"?
input_20?????????
.
input_21"?
input_21?????????
.
input_22"?
input_22?????????
.
input_23"?
input_23?????????
.
input_24"?
input_24?????????
.
input_25"?
input_25?????????
.
input_26"?
input_26?????????
.
input_27"?
input_27?????????
.
input_28"?
input_28?????????
.
input_29"?
input_29?????????
,
input_3!?
input_3?????????
.
input_30"?
input_30?????????
.
input_31"?
input_31?????????
.
input_32"?
input_32?????????
.
input_33"?
input_33?????????
.
input_34"?
input_34?????????
.
input_35"?
input_35?????????
.
input_36"?
input_36?????????
.
input_37"?
input_37?????????
.
input_38"?
input_38?????????
.
input_39"?
input_39?????????
,
input_4!?
input_4?????????
.
input_40"?
input_40?????????
.
input_41"?
input_41?????????
.
input_42"?
input_42?????????
.
input_43"?
input_43?????????
.
input_44"?
input_44?????????
.
input_45"?
input_45?????????
.
input_46"?
input_46?????????
.
input_47"?
input_47?????????
.
input_48"?
input_48?????????
.
input_49"?
input_49?????????
,
input_5!?
input_5?????????
.
input_50"?
input_50?????????
,
input_6!?
input_6?????????
,
input_7!?
input_7?????????
,
input_8!?
input_8?????????
,
input_9!?
input_9?????????"3?0
.
output_1"?
output_1?????????