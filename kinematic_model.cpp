#include "kinematic_model.hpp"

#include <rl/mdl/UrdfFactory.h>
#include <rl/mdl/Kinematic.h>
#include <rl/mdl/Body.h>
#include <rl/mdl/NloptInverseKinematics.h>
#include <rl/mdl/Prismatic.h>
#include <rl/math/Transform.h>
#include <rl/math/Quaternion.h>

#include <random>
#include <memory>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <spdlog/spdlog.h>

static constexpr unsigned int kIkRandomSeed = 42;

namespace hdv::robot {

class KinematicModel::Impl {
public:
    explicit Impl(hdv::PoseInterpreterConfig const& config,
                  std::filesystem::path const& urdfPath,
                  std::shared_ptr<rl::mdl::Kinematic> model)
        : m_config(config)
        , m_urdfPath(urdfPath)
        , m_model(std::move(model))
    {
        m_poseInterpreter = hdv::createInterpreter(config);
    }

    explicit Impl(hdv::PoseInterpreterConfig const& config,
                  std::shared_ptr<rl::mdl::Kinematic> model)
        : m_config(config)
        , m_model(std::move(model))
    {
        m_poseInterpreter = hdv::createInterpreter(config);
    }

    hdv::PoseInterpreterConfig const& config() const { return m_config; }
    std::shared_ptr<hdv::PoseInterpreter> const& poseInterpreter() const { return m_poseInterpreter; }
    std::filesystem::path const& urdfPath() const { return m_urdfPath; }
    std::shared_ptr<rl::mdl::Kinematic> const& model() const { return m_model; }

    void setConfig(hdv::PoseInterpreterConfig const& cfg)
    {
        m_config = cfg;
        m_poseInterpreter = hdv::createInterpreter(cfg);
    }

private:
    hdv::PoseInterpreterConfig                  m_config;
    std::shared_ptr<hdv::PoseInterpreter>       m_poseInterpreter;
    std::filesystem::path                       m_urdfPath;
    std::shared_ptr<rl::mdl::Kinematic>         m_model;
};


void KinematicModel::ImplDeleter::operator()(Impl* p) noexcept { delete p; }

KinematicModel::~KinematicModel() = default;



KinematicModel::KinematicModel(KinematicModel&& other) noexcept
    : m_impl(std::move(other.m_impl))
{}

KinematicModel& KinematicModel::operator=(KinematicModel&& other) noexcept
{
    if (this != &other)
        m_impl = std::move(other.m_impl);
    return *this;
}


KinematicModel KinematicModel::load(
    std::filesystem::path const& urdfFilePath,
    hdv::PoseInterpreterConfig const& config)
{
    rl::mdl::UrdfFactory factory;
    auto model = std::dynamic_pointer_cast<rl::mdl::Kinematic>(
        factory.create(urdfFilePath.string())
    );
    if (!model)
        throw std::runtime_error(
            "KinematicModel::load: cannot load URDF: " + urdfFilePath.string());

    KinematicModel kin;
    kin.m_impl = std::unique_ptr<Impl, ImplDeleter>(
        new Impl(config, urdfFilePath, std::move(model))
    );
    return kin;
}


void KinematicModel::save(KinematicModel const& kin, std::filesystem::path const& urdfFilePath)
{
    throw std::logic_error("KinematicModel::save is not implemented");
}

bool KinematicModel::setPoseRepresentation(hdv::PoseInterpreterConfig const& config)
{
    if (!m_impl) return false;
    m_impl->setConfig(config);
    return m_impl->poseInterpreter() != nullptr;
}

hdv::PoseInterpreterConfig const& KinematicModel::getPoseRepresentation() const
{
    static const hdv::PoseInterpreterConfig empty{};
    return m_impl ? m_impl->config() : empty;
}


std::vector<float> KinematicModel::solveFK(
    std::vector<float> const& jointConfiguration_rad, size_t index)
const {
    if (!m_impl || !m_impl->model() || !m_impl->poseInterpreter()) {
        spdlog::error("KinematicModel::solveFK: model or pose interpreter not initialized");
        return {};
    }

    auto& model = m_impl->model();
    auto& interp = m_impl->poseInterpreter();

    if (!isWithinJointLimit(jointConfiguration_rad)) {
        rl::math::Vector minL = model->getMinimum();
        rl::math::Vector maxL = model->getMaximum();
        for (size_t i = 0; i < jointConfiguration_rad.size(); ++i) {
            float val = jointConfiguration_rad[i];
            if (val < static_cast<float>(minL[i]) || val > static_cast<float>(maxL[i])) {
                std::ostringstream oss;
                oss << "KinematicModel::solveFK: joint " << i
                    << " violates limit [" << minL[i] << ", " << maxL[i] << "]"
                    << ", given: " << val;
                spdlog::error(oss.str());
            }
        }
        return {};
    }

    rl::math::Vector q(jointConfiguration_rad.size());
    for (size_t i = 0; i < jointConfiguration_rad.size(); ++i)
        q[i] = jointConfiguration_rad[i];

    model->setPosition(q);
    model->forwardPosition();

    rl::math::Transform rlTf = model->getOperationalPosition(index);

    hdv::EigenTf hdvTf;
    hdvTf.matrix() = rlTf.matrix().cast<float>();
    interp->setTransform(hdvTf);

    return interp->getData();
}


std::optional<std::vector<float>> KinematicModel::solveIK(
    std::vector<float> const& poseValue,
    std::vector<float> initialJoints,
    uint32_t maxNTrials,
    uint32_t timeout_ms)
const {
    if (!m_impl || !m_impl->model() || !m_impl->poseInterpreter()) {
        spdlog::error("KinematicModel::solveIK: model or pose interpreter not initialized");
        return std::nullopt;
    }

    auto& model = m_impl->model();
    auto& interp = m_impl->poseInterpreter();

    interp->setData(poseValue);
    hdv::EigenTf hdvGoal = interp->getTransform();

    rl::math::Transform goal;
    goal.matrix() = hdvGoal.matrix().cast<double>();

    rl::mdl::NloptInverseKinematics ik(model.get());

    ik.setDuration(std::chrono::milliseconds(timeout_ms));
    ik.addGoal(goal, 0);

    if (!initialJoints.empty()) {
        rl::math::Vector qInit(initialJoints.size());
        for (size_t i = 0; i < initialJoints.size(); ++i)
            qInit[i] = initialJoints[i];
        model->setPosition(qInit);
    }

    if (ik.solve()) {
        rl::math::Vector q = model->getPosition();
        std::vector<float> result(q.data(), q.data() + q.size());
        if (!isWithinJointLimit(result)) {
            spdlog::warn("KinematicModel::solveIK: solution violates joint limits");
            return std::nullopt;
        }
        return result;
    }

    int dof = model->getDof();
    rl::math::Vector minL = model->getMinimum();
    rl::math::Vector maxL = model->getMaximum();
    std::mt19937 rng(kIkRandomSeed);

    for (int attempt = 0; attempt < maxNTrials; ++attempt) {
        rl::math::Vector qRand(dof);
        for (int j = 0; j < dof; ++j) {
            std::uniform_real_distribution<double> dist(minL(j), maxL(j));
            qRand(j) = dist(rng);
        }
        model->setPosition(qRand);
        if (ik.solve()) {
            rl::math::Vector q = model->getPosition();
            std::vector<float> result(q.data(), q.data() + q.size());
            if (!isWithinJointLimit(result)) continue;
            return result;
        }
    }

    spdlog::warn("KinematicModel::solveIK: no solution found after {} random-restart attempts",
                 maxNTrials);
    return std::nullopt;
}

std::vector<std::vector<float>> KinematicModel::solveIKMultiple(
    std::vector<float> const& poseValue, uint32_t maxSolutions, uint32_t timeout_ms)
const {
    if (!m_impl || !m_impl->model() || !m_impl->poseInterpreter()) {
        spdlog::error("KinematicModel::solveIKMultiple: model or pose interpreter not initialized");
        return {};
    }

    auto& model = m_impl->model();
    auto& interp = m_impl->poseInterpreter();

    interp->setData(poseValue);
    hdv::EigenTf hdvGoal = interp->getTransform();

    rl::math::Transform goal;
    goal.matrix() = hdvGoal.matrix().cast<double>();

    rl::mdl::NloptInverseKinematics ik(model.get());

    ik.setDuration(std::chrono::milliseconds(timeout_ms));
    ik.addGoal(goal, 0);

    int dof = model->getDof();
    rl::math::Vector minL = model->getMinimum();
    rl::math::Vector maxL = model->getMaximum();
    std::mt19937 rng(kIkRandomSeed);

    std::vector<std::vector<float>> solutions;

    for (int attempt = 0; attempt < 50 && (int)solutions.size() < maxSolutions; ++attempt) {
        rl::math::Vector qRand(dof);
        for (int j = 0; j < dof; ++j) {
            std::uniform_real_distribution<double> dist(minL(j), maxL(j));
            qRand(j) = dist(rng);
        }
        model->setPosition(qRand);

        if (ik.solve()) {
            rl::math::Vector q = model->getPosition();
            std::vector<float> result(q.data(), q.data() + q.size());
            if (!isWithinJointLimit(result)) {
                spdlog::debug("KinematicModel::solveIKMultiple: attempt {} found solution outside joint limits", attempt);
                continue;
            }
            solutions.push_back(result);
        }
    }

    if (solutions.empty()) {
        spdlog::warn("KinematicModel::solveIKMultiple: no solution found after 50 attempts");
    } else {
        spdlog::debug("KinematicModel::solveIKMultiple: found {}/{} solutions",
                      solutions.size(), maxSolutions);
    }
    return solutions;
}


std::vector<float> KinematicModel::alternativeConfiguration(
    std::vector<float> const& joints)
const {
    auto currentPose = solveFK(joints);

    for (int attempt = 0; attempt < 5; ++attempt) {
        auto result = solveIK(currentPose, {});
        if (!result.has_value())
            break;

        bool sameConfig = (result->size() == joints.size());
        if (sameConfig) {
            for (size_t i = 0; i < joints.size(); ++i) {
                if (std::abs((*result)[i] - joints[i]) > 1e-4f) {
                    sameConfig = false;
                    break;
                }
            }
        }

        if (!sameConfig)
            return *result;
    }

    spdlog::warn("KinematicModel::alternativeConfiguration: "
                 "no distinct configuration found, returning original");
    return joints;
}


bool KinematicModel::isWithinJointLimit(std::vector<float> const& joints)
const {
    if (!m_impl || !m_impl->model()) return false;
    auto& model = m_impl->model();

    rl::math::Vector q(joints.size());
    for (size_t i = 0; i < joints.size(); ++i) q[i] = joints[i];
    rl::math::Vector minL = model->getMinimum();
    rl::math::Vector maxL = model->getMaximum();
    for (int i = 0; i < q.size(); ++i)
        if (q[i] < minL[i] || q[i] > maxL[i]) return false;
    return true;
}



int KinematicModel::getDof() const
{
    if (!m_impl || !m_impl->model()) return 0;
    return m_impl->model()->getDof();
}

std::vector<std::string> KinematicModel::frames() const {
    if (!m_impl || !m_impl->model()) return {};
    auto& model = m_impl->model();

    std::vector<std::string> result;
    for (std::size_t i = 0; i < model->getBodies(); ++i)
        result.push_back(model->getBody(i)->getName());
    return result;
}

std::filesystem::path const& KinematicModel::urdfPath() const
{
    static const std::filesystem::path empty{};
    return m_impl ? m_impl->urdfPath() : empty;
}

std::vector<std::pair<std::string, std::vector<float>>>
KinematicModel::computeBodyFrames(
    std::vector<float> const& joints, size_t tcpIndex)
const {
    if (!m_impl || !m_impl->model() || !m_impl->poseInterpreter()) {
        spdlog::error("KinematicModel::computeBodyFrames: model or pose interpreter not initialized");
        return {};
    }

    auto& model = m_impl->model();
    auto& interp = m_impl->poseInterpreter();

    rl::math::Vector q(joints.size());
    for (size_t i = 0; i < joints.size(); ++i) q[i] = joints[i];
    model->setPosition(q);
    model->forwardPosition();

    hdv::PoseInterpreterConfig quatConfig;
    quatConfig.rotationType    = hdv::ROTATION_TYPE::QUATERNION;
    quatConfig.translationUnit = hdv::TRANSLATION_UNIT::METER;
    quatConfig.rotationUnit    = hdv::ROTATION_UNIT::RADIANS;
    quatConfig.rotationVariant = hdv::QUATERNION_TYPE::WXYZ;
    auto quatInterpreter = hdv::createInterpreter(quatConfig);

    std::vector<std::pair<std::string, std::vector<float>>> result;

    for (std::size_t i = 0; i < model->getBodies(); ++i) {
        const auto& bf = model->getBodyFrame(i);
        auto p = bf.translation();
        rl::math::Quaternion r(bf.linear());


        quatInterpreter->setData({
            static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()),
            static_cast<float>(r.w()), static_cast<float>(r.x()), static_cast<float>(r.y()), static_cast<float>(r.z())
        });


        quatInterpreter->convertTo(interp.get());

        result.push_back({
            model->getBody(i)->getName(),
            interp->getData()
        });
    }

    auto tcpT = model->getOperationalPosition(tcpIndex);
    auto tcpP = tcpT.translation();
    rl::math::Quaternion tcpR(tcpT.linear());

    quatInterpreter->setData({
        static_cast<float>(tcpP.x()), static_cast<float>(tcpP.y()), static_cast<float>(tcpP.z()),
        static_cast<float>(tcpR.w()), static_cast<float>(tcpR.x()), static_cast<float>(tcpR.y()), static_cast<float>(tcpR.z())
    });
    quatInterpreter->convertTo(interp.get());

    result.push_back({
        "tcp",
        interp->getData()
    });

    return result;
}

} 