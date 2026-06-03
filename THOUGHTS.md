To answer the problem of  KDD×Tencent Advertising Algorithm Compeetition,there are something you could do to upgrade AUC.
1. Conditional Diffusion may lose its effect in non-multimodal situations,thus you can use Manifold Aggregation
 or other model to replace it.
2. The model feature selection is absense in the model,we could add MMoE orr other fusion gate.
3. To capture the Sequence, we could try to use Longer style model to replace it.
4. Multi-loss training is a hard problem in large model training, try to summarize 
   the loss formula in the same Momentum direction.
5. ProtoType capture and CrossFieldNet may lose deep interaction in CVR task, try to reassign it in MetaFormer style
,without losing or trying to minimize the original idea of the net itself.
6. MetaAlign needs to reconstruct the form,or justt delete it.The purpose of MetaAlign is to adjust the training
effect.However, in first round of the KDD, it interupts the  training jobs. You could try to adjust it.

Anyway,the model might not be the best selection to KDD problem. The problem mentioned in paper hope you'll like it,to 
grasp the meaningfully interacts the inteeractions and sequence model.
