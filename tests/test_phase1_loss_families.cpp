/**
 * @file test_phase1_loss_families.cpp
 * @brief Faz 1: NexusLoss'un tum ailelerinde backward_into sozlesmesi.
 *
 * Her durumlu kayip sinifi icin dort iddia denenir:
 *   1. forward degeri ve turev sonludur,
 *   2. forward'dan sonra `backward_into` (ve `nexus::backward_into(loss, vector&)`)
 *      hicbir heap ayirmasi yapmaz,
 *   3. `backward()` ile `backward_into` bit duzeyinde ayni sonucu verir,
 *   4. analitik turev merkezi farkli sayisal turevle tolerans icinde uyusur.
 * Girdiler parcali dogrusal kayiplarin kirilma noktalarindan (hinge, ReLU,
 * siralama esitligi, en yakin komsu esitligi) bilincli olarak uzak secilmistir.
 *
 * Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
 * 1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
 * olarak gelistirilmistir.
 */

#include "nexus/nexus.hpp"
#include "nexus_test.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nexus_test::AllocationScope;
using nexus_test::expect;
using Span = std::span<const double>;
using Vec = std::vector<double>;
using Indices = std::vector<std::size_t>;

constexpr double kAbsTol = 1.0e-6;
constexpr double kRelTol = 1.0e-5;

Vec concat(std::initializer_list<Vec> parts) {
  Vec joined;
  for (const Vec& part : parts) {
    joined.insert(joined.end(), part.begin(), part.end());
  }
  return joined;
}

std::string label(std::string_view family, std::string_view what) {
  std::string text(family);
  text += ": ";
  text += what;
  return text;
}

std::string gradient_label(std::string_view family, double ratio) {
  std::ostringstream text;
  text << family << ": sayisal turevle uyumlu (hata/tolerans=" << std::setprecision(2) << ratio << ")";
  return text.str();
}

/**
 * @brief Durumlu bir kayip sinifinin dort iddiasini dener.
 * @param forward `double(Loss&, Span x)`: x'i kaybin turevlenen girdilerine dagitir.
 */
template <typename Loss, typename Forward>
void check_stateful(std::string_view name, Loss loss, const Vec& x, Forward forward, double atol = kAbsTol,
                    double rtol = kRelTol) {
  static_assert(nexus::InplaceBackward<Loss, double>, "kayip backward_into(span<double>) sunmali");
  static_assert(nexus::SizedGradient<Loss>, "kayip gradient_size() sunmali");

  const double value = forward(loss, Span(x));
  const Vec reference = loss.backward();
  Vec gradient(loss.gradient_size());
  Vec reusable;
  (void)nexus::backward_into(loss, reusable);
  std::uint64_t allocations = 0;
  {
    AllocationScope scope;
    for (int repeat = 0; repeat < 4; ++repeat) {
      loss.backward_into(std::span<double>(gradient));
      (void)nexus::backward_into(loss, reusable);
    }
    allocations = scope.count();
  }

  Loss probe = loss;
  const Vec numeric = nexus_test::numeric_gradient(Span(x), [&](Span point) { return forward(probe, point); });
  const double ratio =
      gradient.size() == numeric.size() ? nexus_test::tolerance_ratio(gradient, numeric, atol, rtol) : HUGE_VAL;

  expect(std::isfinite(value) && nexus_test::finite<double>(gradient), label(name, "deger ve turev sonlu"));
  expect(allocations == 0, label(name, "backward_into 0 heap ayirma"));
  expect(nexus_test::identical<double>(reference, gradient) && nexus_test::identical<double>(reusable, gradient),
         label(name, "backward == backward_into (bit duzeyinde)"));
  expect(ratio <= 1.0, gradient_label(name, ratio));
}

/**
 * @brief Girdileri her cagrida acikca alan (durumsuz) kayiplar: GAN aileleri.
 * @param value `double(Span x)`, @param gradient_into `void(Span x, std::span<double> grad)`.
 */
template <typename Value, typename GradientInto>
void check_stateless(std::string_view name, const Vec& x, Value value, GradientInto gradient_into,
                     double atol = kAbsTol, double rtol = kRelTol) {
  Vec gradient(x.size());
  gradient_into(Span(x), std::span<double>(gradient));
  std::uint64_t allocations = 0;
  {
    AllocationScope scope;
    gradient_into(Span(x), std::span<double>(gradient));
    allocations = scope.count();
  }
  const Vec numeric = nexus_test::numeric_gradient(Span(x), value);
  const double ratio = nexus_test::tolerance_ratio(gradient, numeric, atol, rtol);
  expect(std::isfinite(value(Span(x))) && nexus_test::finite<double>(gradient), label(name, "deger ve turev sonlu"));
  expect(allocations == 0, label(name, "*_backward_into 0 heap ayirma"));
  expect(ratio <= 1.0, gradient_label(name, ratio));
}

/// LossBase kapisi: forward(pred, target) vektor dondurur, skaler ilk elemandir.
auto against(const Vec& target) {
  return [&target](auto& loss, Span prediction) { return loss.forward(prediction, Span(target))[0]; };
}

void test_regression() {
  nexus_test::section("regresyon (LossBase)");
  const Vec pred{0.3, -1.2, 2.5, 0.8};
  const Vec target{0.1, -0.4, 1.0, 2.0};
  const Vec positive_pred{0.3, 1.2, 2.5, 0.8};
  const Vec positive_target{0.1, 0.4, 1.0, 2.0};
  const Vec counts{1.0, 0.0, 3.0, 2.0};
  const Vec tweedie_target{0.1, 0.0, 1.0, 2.0};

  check_stateful("MSE", nexusloss::MSELoss<double>(), pred, against(target));
  check_stateful("MSE sum", nexusloss::MSELoss<double>(nexusloss::core::ReductionType::Sum), pred, against(target));
  check_stateful("MAE", nexusloss::MAELoss<double>(), pred, against(target));
  check_stateful("Huber", nexusloss::HuberLoss<double>(1.0), pred, against(target));
  check_stateful("LogCosh", nexusloss::LogCoshLoss<double>(), pred, against(target));
  check_stateful("MSLE", nexusloss::MSLELoss<double>(), positive_pred, against(positive_target));
  check_stateful("Quantile", nexusloss::QuantileLoss<double>(0.3), pred, against(target));
  check_stateful("Poisson", nexusloss::PoissonLoss<double>(), pred, against(counts));
  check_stateful("Tweedie", nexusloss::TweedieLoss<double>(1.5), positive_pred, against(tweedie_target));
  check_stateful("Cauchy", nexusloss::CauchyLoss<double>(1.0), pred, against(target));
  check_stateful("Charbonnier", nexusloss::CharbonnierLoss<double>(1.0e-3), pred, against(target));
  check_stateful("TukeyBiweight", nexusloss::TukeyBiweightLoss<double>(4.685), pred, against(target));
}

void test_classification() {
  nexus_test::section("siniflandirma");
  const Vec probabilities{0.2, 0.7, 0.9, 0.4};
  const Vec labels{0.0, 1.0, 1.0, 0.0};
  const Vec logits{0.3, -1.2, 2.5, 0.8};
  const Vec signed_pred{0.3, -0.2, 2.5, 0.8};
  const Vec signs{1.0, -1.0, 1.0, -1.0};

  check_stateful("BCE", nexusloss::BCELoss<double>(), probabilities, against(labels));
  check_stateful("BCEWithLogits", nexusloss::BCEWithLogitsLoss<double>(), logits, against(labels));
  check_stateful("Hinge", nexusloss::HingeLoss<double>(), signed_pred, against(signs));
  check_stateful("SquaredHinge", nexusloss::SquaredHingeLoss<double>(), signed_pred, against(signs));
  check_stateful("Exponential", nexusloss::ExponentialLoss<double>(), signed_pred, against(signs));
  check_stateful("Perceptron", nexusloss::PerceptronLoss<double>(), signed_pred, against(signs));
  check_stateful("ZeroOne", nexusloss::ZeroOneLoss<double>(), signed_pred, against(signs));
  check_stateful("Focal", nexusloss::FocalLoss<double>(0.25, 2.0), probabilities, against(labels));

  const Vec class_logits{0.5, -1.0, 2.0, 0.3};
  const auto to_class = [](auto& loss, Span x) { return loss.forward(x, 2); };
  check_stateful("CrossEntropy", nexusloss::classification::CrossEntropyLoss<double>(0.0), class_logits, to_class);
  check_stateful("CrossEntropy smoothing", nexusloss::classification::CrossEntropyLoss<double>(0.1), class_logits,
                 to_class);
  check_stateful("LabelSmoothingCE", nexusloss::classification::LabelSmoothingCrossEntropyLoss<double>(0.1),
                 class_logits, to_class);

  const Vec kl_target{0.25, 0.25, 0.3, 0.2};
  check_stateful("KLDiv", nexusloss::classification::KLDivLoss<double>(), Vec{0.2, 0.3, 0.4, 0.1},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(kl_target)); });
}

void test_segmentation() {
  namespace seg = nexusloss::segmentation;
  nexus_test::section("segmentasyon");
  const Vec soft{0.8, 0.3, 0.6, 0.1, 0.9};
  const Vec mask{1.0, 0.0, 1.0, 0.0, 1.0};
  const auto with_mask = [&](auto& loss, Span x) { return loss.forward(x, Span(mask)); };
  check_stateful("Dice", seg::DiceLoss<double>(1.0), soft, with_mask);
  check_stateful("SoftIoU", seg::IoULoss<double>(1.0), soft, with_mask);
  check_stateful("Tversky", seg::TverskyLoss<double>(0.3, 0.7, 1.0), soft, with_mask);
  check_stateful("FocalTversky", seg::FocalTverskyLoss<double>(0.3, 0.7, 4.0 / 3.0), soft, with_mask);

  // Sinif basina hatalar birbirinden en az 0.01 uzak; siralama turev boyunca sabit kalir.
  const Vec probabilities{0.72, 0.18, 0.10, 0.14, 0.61, 0.25, 0.33, 0.21, 0.46, 0.52, 0.31, 0.17};
  const Indices pixel_classes{0, 1, 2, 1};
  check_stateful("LovaszSoftmax", seg::LovaszSoftmaxLoss<double>(), probabilities,
                 [&](auto& loss, Span x) { return loss.forward(x, pixel_classes, 4, 3); });

  const Vec hinge_logits{0.8, -0.3, 1.7, -1.1, 0.25};
  const Vec hinge_target{1.0, 0.0, 1.0, 1.0, 0.0};
  check_stateful("LovaszHinge", seg::LovaszHingeLoss<double>(), hinge_logits,
                 [&](auto& loss, Span x) { return loss.forward(x, Span(hinge_target)); });

  // Lovasz uzantisi 0/1 hata vektorunde Jaccard kaybinin kendisidir.
  const Indices all_background{0, 0, 0};
  const Vec complete_miss{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
  const Vec one_hit{0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
  const double miss = seg::lovasz_softmax_loss<double>(complete_miss, all_background, 3, 2);
  const double hit = seg::lovasz_softmax_loss<double>(one_hit, all_background, 3, 2);
  expect(std::fabs(miss - 1.0) < 1.0e-12, "LovaszSoftmax: tam iskalamada Jaccard kaybi 1");
  expect(std::fabs(hit - 2.0 / 3.0) < 1.0e-12, "LovaszSoftmax: 3 pikselde 1 isabet -> 2/3");
  const Vec confident{4.0, -3.0, 5.0};
  const Vec confident_target{1.0, 0.0, 1.0};
  expect(seg::lovasz_hinge_loss<double>(confident, confident_target) == 0.0,
         "LovaszHinge: tum marjlar >= 1 ise kayip 0");
}

void test_detection() {
  namespace det = nexusloss::detection;
  nexus_test::section("tespit");
  const Vec logits{0.3, -1.2, 2.5, 0.8};
  const Vec labels{0.0, 1.0, 1.0, 0.0};
  check_stateful("DetectionFocal", det::DetectionFocalLoss<double>(0.25, 2.0), logits,
                 [&](auto& loss, Span x) { return loss.forward(x, Span(labels)); });
  const Vec target{0.1, -0.4, 1.0, 2.0};
  check_stateful("SmoothL1BBox", det::SmoothL1BBoxLoss<double>(1.0), logits,
                 [&](auto& loss, Span x) { return loss.forward(x, Span(target)); });

  // Iki kutu; kesisim ve kapsayan kutu kenarlari farkli kutulara ait ve en az 0.1 ayrik.
  const Vec predicted_boxes{0.1, 0.2, 2.3, 1.9, 1.0, 1.1, 3.2, 2.5};
  const Vec target_boxes{0.5, 0.0, 2.0, 2.4, 0.8, 1.4, 2.9, 3.0};
  const auto with_boxes = [&](auto& loss, Span x) { return loss.forward(x, Span(target_boxes)); };
  check_stateful("IoU [N,4]", det::IoULoss<double>(), predicted_boxes, with_boxes);
  check_stateful("GIoU [N,4]", det::GIoULoss<double>(), predicted_boxes, with_boxes);
  check_stateful("DIoU [N,4]", det::DIoULoss<double>(), predicted_boxes, with_boxes);
  check_stateful("CIoU [N,4]", det::CIoULoss<double>(), predicted_boxes, with_boxes);
}

void test_metric() {
  namespace metric = nexusloss::metric;
  nexus_test::section("metrik ogrenme");
  const Vec a{0.3, -0.2, 0.5};
  const Vec b{0.1, 0.4, -0.3};
  const Vec negative{0.9, 0.1, 0.2};
  check_stateful("Contrastive farkli", metric::ContrastiveLoss<double>(2.0), concat({a, b}),
                 [](auto& loss, Span x) { return loss.forward(x.first(3), x.subspan(3), false); });
  check_stateful("Contrastive ayni", metric::ContrastiveLoss<double>(2.0), concat({a, b}),
                 [](auto& loss, Span x) { return loss.forward(x.first(3), x.subspan(3), true); });
  check_stateful("Triplet", metric::TripletLoss<double>(1.0), concat({a, b, negative}),
                 [](auto& loss, Span x) { return loss.forward(x.first(3), x.subspan(3, 3), x.subspan(6)); });

  const Vec cosine{0.2, 0.6, -0.3, 0.1};
  const auto to_target = [](auto& loss, Span x) { return loss.forward(x, 1); };
  check_stateful("ArcFace", metric::ArcFaceLoss<double>(0.5, 4.0), cosine, to_target);
  check_stateful("CosFace", metric::CosFaceLoss<double>(0.35, 4.0), cosine, to_target);
  check_stateful("SphereFace", metric::SphereFaceLoss<double>(1.0, 4.0), cosine, to_target);
}

void test_ranking() {
  namespace rank = nexusloss::ranking;
  nexus_test::section("siralama");
  check_stateful("MarginRanking", rank::MarginRankingLoss<double>(0.1), Vec{0.2, 0.5},
                 [](auto& loss, Span x) { return loss.forward(x[0], x[1], 1.0); });
  check_stateful("RankNet", rank::RankNetLoss<double>(), Vec{0.7, 0.2},
                 [](auto& loss, Span x) { return loss.forward(x[0], x[1]); });
  const Vec scores{0.5, -0.3, 1.2, 0.1};
  const Vec relevance{2.0, 0.0, 3.0, 1.0};
  const auto with_relevance = [&](auto& loss, Span x) { return loss.forward(x, Span(relevance)); };
  check_stateful("ListNet", rank::ListNetLoss<double>(), scores, with_relevance);
  check_stateful("ListMLE O(n)", rank::ListMLELoss<double>(), scores, with_relevance);

  // ListMLE = sum_i [logsumexp(s_{o_i..n}) - s_{o_i}], o = alakaya gore azalan sira.
  const Indices order{2, 0, 3, 1};
  double reference = 0.0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    double tail = 0.0;
    for (std::size_t j = i; j < order.size(); ++j) {
      tail += std::exp(scores[order[j]]);
    }
    reference += std::log(tail) - scores[order[i]];
  }
  rank::ListMLELoss<double> listmle;
  expect(std::fabs(listmle.forward(scores, relevance) - reference) < 1.0e-12,
         "ListMLE O(n) sonek log-sum-exp kapali formulle ayni");
}

void test_generative() {
  namespace gen = nexusloss::generative;
  nexus_test::section("uretken");
  check_stateful("VAE ELBO", gen::VAEELBOLoss<double>(0.7), concat({{0.2, -0.5, 0.9}, {-0.3, 0.4, 0.1}}),
                 [](auto& loss, Span x) { return loss.forward(1.3, x.first(3), x.subspan(3)); });

  const Vec real_fake = concat({{0.8, -0.2, 1.5}, {-0.6, 0.3}});
  const Vec fake{-0.6, 0.3};
  gen::AdversarialLoss<double> vanilla(false);
  gen::AdversarialLoss<double> non_saturating(true);
  check_stateless(
      "GAN D", real_fake, [&](Span x) { return vanilla.discriminator(x.first(3), x.subspan(3)); },
      [&](Span x, std::span<double> g) { vanilla.discriminator_backward_into(x.first(3), x.subspan(3), g); });
  check_stateless(
      "GAN G", fake, [&](Span x) { return vanilla.generator(x); },
      [&](Span x, std::span<double> g) { vanilla.generator_backward_into(x, g); });
  check_stateless(
      "GAN G non-saturating", fake, [&](Span x) { return non_saturating.generator(x); },
      [&](Span x, std::span<double> g) { non_saturating.generator_backward_into(x, g); });

  const Vec critic = concat({{0.4, -0.2, 0.3}, {-0.1, 0.35}});
  gen::WassersteinLoss<double> wasserstein;
  gen::HingeGANLoss<double> hinge;
  check_stateless(
      "WGAN D", critic, [&](Span x) { return wasserstein.discriminator(x.first(3), x.subspan(3)); },
      [&](Span x, std::span<double> g) { wasserstein.discriminator_backward_into(x.first(3), x.subspan(3), g); });
  check_stateless(
      "WGAN G", fake, [&](Span x) { return wasserstein.generator(x); },
      [&](Span x, std::span<double> g) { wasserstein.generator_backward_into(x, g); });
  check_stateless(
      "HingeGAN D", critic, [&](Span x) { return hinge.discriminator(x.first(3), x.subspan(3)); },
      [&](Span x, std::span<double> g) { hinge.discriminator_backward_into(x.first(3), x.subspan(3), g); });
  check_stateless(
      "HingeGAN G", fake, [&](Span x) { return hinge.generator(x); },
      [&](Span x, std::span<double> g) { hinge.generator_backward_into(x, g); });

  gen::LSGANLoss<double> lsgan;
  check_stateless(
      "LSGAN", Vec{0.4, -0.2, 0.9}, [&](Span x) { return lsgan.forward(x, 1.0); },
      [&](Span x, std::span<double> g) { lsgan.backward_into(x, 1.0, g); });
}

void test_distillation() {
  namespace distill = nexusloss::distillation;
  nexus_test::section("damitma");
  const Vec teacher_logits{1.0, 0.2, -0.5, 0.7};
  check_stateful("KnowledgeDistillation", distill::KnowledgeDistillationLoss<double>(2.0), Vec{0.5, -1.0, 2.0, 0.3},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(teacher_logits)); });
  const Vec teacher_features{0.1, -0.4, 1.0};
  check_stateful("FeatureDistillation", distill::FeatureDistillationLoss<double>(), Vec{0.3, -1.2, 2.5},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(teacher_features)); });
  const Vec teacher_map{0.1, 0.4, 1.0, -2.0};
  check_stateful("AttentionTransfer", distill::AttentionTransferLoss<double>(), Vec{0.3, -1.2, 2.5, 0.8},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(teacher_map)); });
}

void test_self_supervised() {
  namespace ssl = nexusloss::self_supervised;
  nexus_test::section("oz-denetimli");
  const auto positive_then_negatives = [](auto& loss, Span x) { return loss.forward(x[0], x.subspan(1)); };
  check_stateful("InfoNCE", ssl::InfoNCELoss<double>(0.5), Vec{0.6, 0.1, -0.2, 0.4}, positive_then_negatives);
  check_stateful("NT-Xent", ssl::NTXentLoss<double>(0.5), Vec{0.6, 0.1, -0.2, 0.4}, positive_then_negatives);
  const Vec byol_target{0.4, 0.2, -1.0};
  check_stateful("BYOL", ssl::BYOLLoss<double>(), Vec{0.3, -1.2, 2.5},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(byol_target)); });
  const Vec teacher{0.1, 0.2, 0.6, 0.1};
  check_stateful("DINO", ssl::DINOLoss<double>(0.2), Vec{0.5, -1.0, 2.0, 0.3},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(teacher)); });
  check_stateful("BarlowTwins", ssl::BarlowTwinsLoss<double>(0.005),
                 Vec{0.9, 0.1, -0.2, 0.05, 0.8, 0.3, -0.1, 0.2, 1.1},
                 [](auto& loss, Span x) { return loss.forward(x, 3); });

  // Sutun 0'in std'si > gamma (varyans mentesesi kapali), sutun 1 ve 2 < gamma (acik).
  const Vec view_a{1.8, 0.1, -0.5, -1.2, 0.4, 0.3, 0.5, -0.2, 0.9, 2.3, 0.3, -0.7, -1.9, -0.5, 0.2};
  const Vec view_b{1.5, 0.2, -0.3, -1.0, 0.6, 0.1, 0.7, -0.1, 1.1, 2.0, 0.2, -0.9, -1.6, -0.4, 0.4};
  check_stateful("VICReg", ssl::VICRegLoss<double>(), concat({view_a, view_b}),
                 [](auto& loss, Span x) { return loss.forward(x.first(15), x.subspan(15), 5, 3); });
  ssl::VICRegWorkspace<double> workspace;
  const double with_workspace = ssl::vicreg_loss<double>(view_a, view_b, 5, 3, workspace);
  const double legacy = ssl::vicreg_loss<double>(view_a, view_b, 5, 3);
  expect(with_workspace == legacy, "VICReg: workspace ve eski imza ayni deger");
}

void test_sequence() {
  namespace seq = nexusloss::sequence;
  nexus_test::section("dizi");
  Vec ctc_logits(6 * 4);
  for (std::size_t i = 0; i < ctc_logits.size(); ++i) {
    ctc_logits[i] = 1.5 * std::sin(0.7 * static_cast<double>(i) + 0.3);
  }
  const Indices labels{1, 2, 2};
  check_stateful("CTC (tekrarli etiket)", seq::CTCLoss<double>(), ctc_logits,
                 [&](auto& loss, Span x) { return loss.forward(x, 6, 4, labels, 0); });

  Vec token_logits(4 * 5);
  for (std::size_t i = 0; i < token_logits.size(); ++i) {
    token_logits[i] = std::cos(0.9 * static_cast<double>(i));
  }
  const Indices positions{1, 3, 1};
  const Indices targets{2, 0, 4};
  check_stateful("MaskedSequenceCE (yinelenen konum)", seq::MaskedSequenceCrossEntropyLoss<double>(), token_logits,
                 [&](auto& loss, Span x) { return loss.forward(x, 4, 5, positions, targets); });
}

void test_survival() {
  namespace surv = nexusloss::survival;
  nexus_test::section("sagkalim");
  const Vec durations{5.0, 3.0, 8.0, 1.0, 6.0};
  const bool observed[5] = {true, false, true, true, false};
  check_stateful("CoxPH", surv::CoxPHLoss<double>(), Vec{0.3, -0.5, 1.1, 0.2, -0.8},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(durations), std::span<const bool>(observed)); });
  check_stateful("Weibull olay", surv::WeibullNLLLoss<double>(), Vec{0.3, -0.2},
                 [](auto& loss, Span x) { return loss.forward(2.5, true, x[0], x[1]); });
  check_stateful("Weibull sansur", surv::WeibullNLLLoss<double>(), Vec{0.3, -0.2},
                 [](auto& loss, Span x) { return loss.forward(2.5, false, x[0], x[1]); });
}

void test_point_cloud() {
  namespace pc = nexusloss::point_cloud;
  nexus_test::section("nokta bulutu");
  const Vec a{0.0, 0.1, 0.2, 1.0, 0.9, 1.1, -0.8, 0.5, 0.3};
  const Vec b{0.2, 0.0, 0.1, 1.2, 1.0, 0.8, -0.5, 0.7, 0.2, 0.4, -0.6, 0.9};
  check_stateful("Chamfer", pc::ChamferDistanceLoss<double>(3), concat({a, b}),
                 [](auto& loss, Span x) { return loss.forward(x.first(9), x.subspan(9)); });
  const Vec b_equal(b.begin(), b.begin() + 9);
  check_stateful("EMD (Hungarian)", pc::EarthMoverDistanceLoss<double>(3), concat({a, b_equal}),
                 [](auto& loss, Span x) { return loss.forward(x.first(9), x.subspan(9)); });
}

void test_reinforcement() {
  namespace rl = nexusloss::reinforcement;
  nexus_test::section("pekistirmeli ogrenme");
  const Vec advantages{1.0, -0.5, 2.0};
  check_stateful("PolicyGradient", rl::PolicyGradientLoss<double>(), Vec{-0.5, -1.2, -0.3},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(advantages)); });
  // Oranlar: 1.05 (kirpma ici), 1.49 (A>0, kirpilmis), 0.61 (A<0, kirpilmis).
  const Vec old_logs{-0.5, -1.2, -0.5};
  const Vec ppo_advantages{1.0, 2.0, -1.5};
  check_stateful("PPOClip", rl::PPOClipLoss<double>(0.2), Vec{-0.45, -0.8, -1.0},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(old_logs), Span(ppo_advantages)); });
  const Vec returns{0.1, -0.4, 1.0};
  check_stateful("Value", rl::ValueLoss<double>(), Vec{0.3, -1.2, 2.5},
                 [&](auto& loss, Span x) { return loss.forward(x, Span(returns)); });
  check_stateful("EntropyBonus", rl::EntropyBonusLoss<double>(), Vec{0.2, 0.5, 0.3},
                 [](auto& loss, Span x) { return loss.forward(x); });
}

}  // namespace

int main() {
  std::cout << "Nexus Faz 1 kayip aileleri testi\n";
  try {
    test_regression();
    test_classification();
    test_segmentation();
    test_detection();
    test_metric();
    test_ranking();
    test_generative();
    test_distillation();
    test_self_supervised();
    test_sequence();
    test_survival();
    test_point_cloud();
    test_reinforcement();
  } catch (const std::exception& error) {
    std::cout << "  FAIL  istisna: " << error.what() << "\n";
    return 1;
  }
  return nexus_test::finish("test_phase1_loss_families");
}
