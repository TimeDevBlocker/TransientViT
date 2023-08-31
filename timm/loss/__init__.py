from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .binary_cross_entropy import BinaryCrossEntropy
from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .regression_loss import weighted_mse_loss, weighted_focal_l1_loss, \
                            weighted_focal_mse_loss, weighted_huber_loss, weighted_l1_loss, weighted_l1_dex_loss, coral_loss, loss_conditional_v2, MeanVarianceLoss
from .jsd import JsdCrossEntropy
