// pi0.5 state prompt, OpenPI normalization, drive-plan encoding/renewal rules, and the
// producer against a real kaivuri-control in --simulate-imu mode (dry PWM, no hardware).
#include "kaivuri/pi05_policy.hpp"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
using namespace kaivuri;
using namespace kaivuri::pi05;
namespace fs = std::filesystem;
static int checks = 0;
static void expect(bool v, const std::string &label) {
    ++checks;
    if (!v)
        throw std::runtime_error(label);
}
template <class F> static void expect_throw(F f, const std::string &label) {
    bool threw = false;
    try {
        f();
    } catch (const std::exception &) {
        threw = true;
    }
    expect(threw, label + " must throw");
}
static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
// Generated with numpy 1.26 (float32 rad2deg, float64 OpenPI quantile normalization,
// np.digitize over linspace(-1, 1, 257)[:-1]); q01/q99 below.
static const char *radian_text[][3] = {
    {"0.800458240", "2.106626245", "1.523291313"},
    {"-1.119005488", "-0.759201832", "1.993056538"},
    {"-2.174726538", "1.741896408", "1.625933258"},
    {"0.046087774", "-0.745444351", "-0.863557062"},
    {"-0.976625979", "-0.063633732", "0.221831643"},
    {"0.456787290", "2.578401360", "1.604777212"},
    {"0.786460301", "2.547008709", "-1.166518248"},
    {"-1.430982237", "0.740190101", "-1.989078362"},
    {"-2.028734662", "0.271466337", "0.037788922"},
    {"2.202405311", "0.820286022", "0.267764704"},
    {"0.184992490", "-1.011928374", "-2.143388677"},
    {"-1.276469709", "1.121754180", "-1.237087725"},
    {"-0.426225709", "-2.182075638", "1.784229103"},
    {"-1.458586811", "-0.915523338", "2.025594339"},
    {"0.246995887", "1.866321183", "0.870642401"},
    {"1.360500547", "-1.760821096", "0.397490343"},
    {"0.237306734", "1.982429008", "-0.465932517"},
    {"0.671283523", "-1.915592117", "-0.339367355"},
    {"-0.649425538", "-1.479041300", "1.718422898"},
    {"-0.378658377", "2.497989845", "0.631960126"},
    {"0.704270018", "0.862383588", "1.046961170"},
    {"-1.476217508", "-0.086495357", "-1.050092983"},
    {"-0.268008169", "-1.735820349", "2.445574645"},
    {"-1.167980621", "1.024472781", "-0.757983609"},
    {"1.995569726", "0.978630744", "-1.568244084"},
    {"1.856356740", "2.335751221", "2.138800583"},
    {"0.534651910", "-1.501792222", "-1.276175224"},
    {"2.253947287", "0.451167141", "-1.333348007"},
    {"2.043473092", "0.879544185", "0.534532517"},
    {"-0.393818387", "-0.227414646", "-1.050451779"},
    {"-2.017325024", "2.005850279", "0.045105041"},
    {"0.428648956", "-0.653616113", "1.406359615"},
    {"-2.079055020", "-0.413510692", "-2.054318587"},
    {"-1.610117909", "2.442311530", "0.957251504"},
    {"-0.144542817", "0.313952518", "1.989484201"},
    {"-0.547788798", "0.633396715", "1.081684992"},
    {"-0.494013870", "0.291672735", "1.473187440"},
    {"2.164060707", "-1.474901065", "2.280413083"},
    {"-2.175141444", "1.414292017", "1.690528786"},
    {"-1.543532553", "-0.189262476", "1.713230134"},
    {"-2.131498290", "0.816617350", "1.606513559"},
    {"0.262417198", "1.284077219", "-1.113167283"},
    {"-1.247098490", "-0.456990637", "-1.338851068"},
    {"-0.538905092", "2.350995495", "0.551997045"},
    {"-0.567673276", "-0.896681825", "2.369789555"},
    {"-0.066504594", "2.505894804", "0.274508812"},
    {"0.301597419", "2.103394514", "1.365283607"},
    {"0.587133746", "-0.152082382", "2.015301725"},
};
static const struct {
    float state[3];
    int bins[3];
} bin_cases[] = {
    {{45.862876892089844f, 120.70079040527344f, 87.27816009521484f}, {214, 255, 255}},
    {{-64.11428833007812f, -43.49905776977539f, 114.19371795654297f}, {-1, -1, 255}},
    {{-124.60264587402344f, 99.80330657958984f, 93.15911102294922f}, {-1, 208, 255}},
    {{2.640634775161743f, -42.710811614990234f, -49.478172302246094f}, {92, -1, 76}},
    {{-55.95654296875f, -3.645944118499756f, 12.710016250610352f}, {-1, -1, 194}},
    {{26.171981811523438f, 147.73150634765625f, 91.94695281982422f}, {158, 255, 255}},
    {{45.06085205078125f, 145.93284606933594f, -66.83656311035156f}, {211, 255, 43}},
    {{-81.9892349243164f, 42.409767150878906f, -113.96578979492188f}, {-1, 74, -1}},
    {{-116.23793029785156f, 15.553874969482422f, 2.1651456356048584f}, {-1, 11, 174}},
    {{126.18851470947266f, 46.998924255371094f, 15.341787338256836f}, {255, 85, 199}},
    {{10.599287986755371f, -57.979217529296875f, -122.80712127685547f}, {114, -1, -1}},
    {{-73.13632202148438f, 64.27177429199219f, -70.87989807128906f}, {-1, 125, 36}},
    {{-24.42093276977539f, -125.02371978759766f, 102.22879791259766f}, {16, -1, 255}},
    {{-83.57086181640625f, -52.45561981201172f, 116.05799102783203f}, {-1, -1, 255}},
    {{14.15182113647461f, 106.93231964111328f, 49.884132385253906f}, {124, 225, 255}},
    {{77.95093536376953f, -100.88761138916016f, 22.774518966674805f}, {255, -1, 213}},
    {{13.596673011779785f, 113.58480834960938f, -26.695964813232422f}, {123, 241, 119}},
    {{38.46171188354492f, -109.75533294677734f, -19.444316864013672f}, {193, -1, 133}},
    {{-37.20934295654297f, -84.7428207397461f, 98.4583740234375f}, {-1, -1, 255}},
    {{-21.695526123046875f, 143.124267578125f, 36.208648681640625f}, {24, 255, 238}},
    {{40.3516960144043f, 49.41093826293945f, 59.986454010009766f}, {198, 90, 255}},
    {{-84.5810317993164f, -4.9558186531066895f, -60.165889739990234f}, {-1, -1, 56}},
    {{-15.35573673248291f, -99.45516967773438f, 140.12110900878906f}, {41, -1, 255}},
    {{-66.92035675048828f, 58.697967529296875f, -43.42926025390625f}, {-1, 112, 88}},
    {{114.33771514892578f, 56.071407318115234f, -89.853759765625f}, {255, 106, 0}},
    {{106.36140441894531f, 133.8286895751953f, 122.54424285888672f}, {255, 255, 255}},
    {{30.633296966552734f, -86.04634857177734f, -73.11945343017578f}, {171, -1, 32}},
    {{129.14166259765625f, 25.849971771240234f, -76.39521026611328f}, {255, 35, 25}},
    {{117.0823745727539f, 50.394168853759766f, 30.626453399658203f}, {255, 93, 227}},
    {{-22.564128875732422f, -13.029898643493652f, -60.18644714355469f}, {21, -1, 56}},
    {{-115.58419799804688f, 114.92675018310547f, 2.5843281745910645f}, {-1, 244, 174}},
    {{24.55977439880371f, -37.44944381713867f, 80.5784683227539f}, {154, -1, 255}},
    {{-119.1210708618164f, -23.692415237426758f, -117.70378112792969f}, {-1, -1, -1}},
    {{-92.2529525756836f, 139.9341278076172f, 54.846466064453125f}, {-1, 255, 255}},
    {{-8.281692504882812f, 17.9881534576416f, 113.98904418945312f}, {61, 17, 255}},
    {{-31.385984420776367f, 36.290958404541016f, 61.97597885131836f}, {-1, 60, 255}},
    {{-28.304908752441406f, 16.71161651611328f, 84.40741729736328f}, {5, 14, 255}},
    {{123.99153137207031f, -84.50559997558594f, 130.6580352783203f}, {255, -1, 255}},
    {{-124.62641143798828f, 81.032958984375f, 96.86016082763672f}, {-1, 164, 255}},
    {{-88.43789672851562f, -10.843940734863281f, 98.16085052490234f}, {-1, -1, 255}},
    {{-122.1258544921875f, 46.788726806640625f, 92.04644012451172f}, {-1, 84, 255}},
    {{15.035396575927734f, 73.57219696044922f, -63.77978515625f}, {127, 147, 49}},
    {{-71.45347595214844f, -26.18363380432129f, -76.71051025390625f}, {-1, -1, 25}},
    {{-30.876985549926758f, 134.70211791992188f, 31.627099990844727f}, {-1, 255, 229}},
    {{-32.5252799987793f, -51.37608337402344f, 135.7789306640625f}, {-1, -1, 255}},
    {{-3.8104326725006104f, 143.5771942138672f, 15.728195190429688f}, {74, 255, 199}},
    {{17.280258178710938f, 120.515625f, 78.22498321533203f}, {133, 255, 255}},
    {{33.64028549194336f, -8.713678359985352f, 115.46827697753906f}, {179, -1, 255}},
    {{-30.250001907348633f, 10.499999046325684f, -90.12500762939453f}, {-1, -1, -1}},
    {{-30.25f, 10.5f, -90.125f}, {0, 0, 0}},
    {{-30.249998092651367f, 10.500000953674316f, -90.12499237060547f}, {0, 0, 0}},
    {{-29.894533157348633f, 10.927733421325684f, -89.59522247314453f}, {0, 0, 0}},
    {{-29.89453125f, 10.927734375f, -89.59521484375f}, {0, 0, 0}},
    {{-29.894529342651367f, 10.927735328674316f, -89.59520721435547f}, {1, 1, 1}},
    {{-17.097658157348633f, 26.326169967651367f, -70.52295684814453f}, {36, 36, 36}},
    {{-17.09765625f, 26.326171875f, -70.52294921875f}, {36, 36, 36}},
    {{-17.097654342651367f, 26.326173782348633f, -70.52294158935547f}, {37, 37, 37}},
    {{14.89453125f, 64.82225799560547f, -22.842287063598633f}, {126, 126, 126}},
    {{14.894532203674316f, 64.822265625f, -22.84228515625f}, {127, 126, 126}},
    {{14.894533157348633f, 64.82227325439453f, -22.842283248901367f}, {127, 127, 127}},
    {{15.25f, 65.24999237060547f, -22.312501907348633f}, {127, 127, 127}},
    {{15.250000953674316f, 65.25f, -22.3125f}, {128, 127, 127}},
    {{15.250001907348633f, 65.25000762939453f, -22.312498092651367f}, {128, 128, 128}},
    {{40.843746185302734f, 96.04686737060547f, 15.83203125f}, {199, 199, 199}},
    {{40.84375f, 96.046875f, 15.832032203674316f}, {199, 199, 200}},
    {{40.843753814697266f, 96.04688262939453f, 15.832033157348633f}, {200, 200, 200}},
    {{60.039058685302734f, 119.14452362060547f, 44.440425872802734f}, {253, 253, 253}},
    {{60.0390625f, 119.14453125f, 44.4404296875f}, {253, 253, 253}},
    {{60.039066314697266f, 119.14453887939453f, 44.440433502197266f}, {254, 254, 254}},
    {{60.394527435302734f, 119.57225799560547f, 44.970211029052734f}, {254, 254, 254}},
    {{60.39453125f, 119.572265625f, 44.97021484375f}, {254, 254, 254}},
    {{60.394535064697266f, 119.57227325439453f, 44.970218658447266f}, {255, 255, 255}},
    {{-1000.0f, 1000.0f, 0.0f}, {-1, 255, 170}},
};
// bins min/max -1 255 count 73
static const struct {
    float x[4];
    double y[4];
} unnormalize_cases[] = {
    {{1.0348999500274658f, -1.0351999998092651f, 0.0f, 1.0f},
     {1.0009289178791896, -0.9712882529838127, 0.032372961557388225, 0.9627046624789236}},
    {{-0.10786130279302597f, 0.6761451363563538f, 0.009327217936515808f, -0.4860824644565582f},
     {-0.10447347178088706, 0.636890549752731, 0.04131320767981972, -0.44572172042750735}},
    {{-0.22867967188358307f, -0.3921617269515991f, 0.005571369547396898f, -0.2626867890357971f},
     {-0.2213420866608672, -0.36701472923831036, 0.0377132051960728, -0.23399973665435925}},
    {{-0.2099452018737793f, -0.9650022983551025f, -0.5647920966148376f, 1.1578402519226074f},
     {-0.2032200932873448, -0.9053223311918903, -0.5089893810930459, 1.1122968855672215}},
    {{-0.46986326575279236f, -0.737865686416626f, 0.23612841963768005f, 0.9850905537605286f},
     {-0.45464101500813714, -0.6918783626739258, 0.25870589242067055, 0.9485742610306106}},
    {{-1.0178170204162598f, -0.14596529304981232f, -0.4424367845058441f, -1.2327135801315308f},
     {-0.9846811850235249, -0.13565992201874666, -0.3917098676894326, -1.1533372463687313}},
};
// Pillow 10.2 Image.resize(BILINEAR) of g[y][x] = (37x + 91y + 5((x y) % 23)) % 256, 47x33.
static const uint8_t pil_16x11[] = {
    112, 116, 124, 140, 132, 122, 135, 131, 115, 123, 143, 141, 143, 140, 132, 110, 115, 131,
    121, 129, 115, 115, 111, 130, 133, 129, 143, 144, 139, 141, 128, 129, 114, 131, 125, 131,
    141, 123, 115, 121, 130, 119, 133, 125, 123, 123, 134, 132, 138, 127, 119, 123, 140, 133,
    129, 145, 142, 123, 138, 125, 127, 134, 120, 147, 136, 147, 98,  128, 118, 119, 120, 145,
    142, 106, 147, 120, 134, 119, 114, 143, 120, 103, 95,  103, 103, 122, 144, 127, 117, 153,
    171, 158, 148, 104, 132, 132, 129, 127, 100, 134, 120, 124, 131, 125, 125, 144, 138, 118,
    105, 88,  110, 113, 112, 143, 104, 145, 138, 104, 128, 121, 141, 117, 124, 135, 100, 138,
    143, 111, 119, 122, 133, 130, 135, 120, 135, 126, 130, 132, 134, 130, 137, 142, 122, 134,
    133, 143, 145, 119, 136, 138, 133, 131, 129, 102, 112, 122, 129, 116, 117, 135, 120, 127,
    136, 119, 132, 141, 129, 131, 139, 130, 121, 108, 125, 119, 130, 136};
static const uint8_t pil_53x7[] = {
    95,  118, 115, 120, 124, 119, 121, 138, 129, 98,  134, 147, 143, 123, 111, 130, 143, 115, 107,
    104, 122, 126, 143, 144, 110, 141, 97,  141, 118, 107, 118, 139, 137, 115, 153, 144, 143, 166,
    137, 119, 142, 164, 141, 109, 137, 173, 154, 122, 121, 119, 111, 137, 96,  120, 113, 115, 121,
    124, 136, 150, 129, 101, 127, 118, 132, 130, 134, 135, 122, 139, 149, 98,  123, 119, 117, 112,
    103, 117, 164, 124, 128, 144, 129, 112, 125, 126, 99,  175, 124, 144, 127, 125, 135, 122, 121,
    161, 78,  171, 113, 113, 143, 146, 124, 123, 155, 117, 151, 124, 141, 142, 142, 144, 128, 98,
    85,  120, 136, 124, 126, 136, 145, 138, 93,  126, 152, 118, 110, 129, 130, 123, 146, 162, 143,
    145, 156, 134, 115, 104, 104, 131, 166, 161, 119, 102, 129, 129, 118, 139, 154, 83,  143, 137,
    120, 105, 118, 124, 126, 165, 147, 125, 120, 134, 131, 112, 105, 114, 98,  69,  104, 108, 107,
    113, 122, 121, 103, 73,  96,  147, 162, 117, 128, 155, 156, 142, 102, 104, 125, 123, 127, 132,
    137, 156, 178, 154, 181, 132, 161, 139, 129, 154, 161, 117, 95,  85,  117, 117, 121, 130, 127,
    119, 135, 137, 115, 105, 122, 134, 136, 148, 142, 66,  86,  132, 143, 154, 143, 137, 141, 129,
    100, 122, 107, 126, 113, 124, 133, 135, 137, 90,  119, 143, 131, 143, 125, 117, 137, 143, 146,
    105, 107, 131, 130, 123, 110, 91,  93,  114, 108, 130, 114, 130, 131, 129, 130, 98,  111, 145,
    115, 117, 119, 122, 136, 149, 138, 117, 135, 116, 128, 134, 136, 138, 136, 133, 121, 103, 139,
    130, 143, 133, 118, 141, 112, 138, 119, 144, 137, 129, 109, 104, 142, 113, 141, 151, 126, 116,
    136, 153, 139, 111, 107, 169, 150, 115, 126, 117, 114, 138, 129, 142, 115, 146, 118, 118, 138,
    135, 128, 175, 140, 112, 132, 99,  109, 136, 135, 130, 169, 111, 138, 145, 150, 133, 114, 118,
    136, 133, 165, 123, 136, 134, 116, 118, 132, 108, 70,  171, 112, 106, 120, 105, 107, 141, 138,
    105, 138, 98,  107, 110, 140, 159, 136, 122, 130};
static const char *status_json(uint64_t imu_ns, const char *a, const char *b, const char *c,
                               bool healthy = true) {
    static std::string text;
    std::ostringstream out;
    out << "{\"state\":\"disarmed\",\"reason\":\"boot\",\"live_pwm\":false,\"imu_healthy\":"
        << (healthy ? "true" : "false")
        << ",\"firmware_blocked\":false,\"imu_host_monotonic_ns\":" << imu_ns
        << ",\"angles_rad\":[0.100000000," << a << "," << b << "," << c
        << "],\"fk_valid\":" << (healthy ? "true" : "false")
        << ",\"enable_slew\":false,\"enable_tracks\":false}";
    text = out.str();
    return text.c_str();
}
// Real PaliGemma pieces from the parent's exporter; heads are stand-ins of the real lengths.
static const char *bundle_yaml = R"({
  "policy": "pi05_openpi_state", "state_dim": 3, "robot_action_dim": 4, "action_dim": 32,
  "prompt_capacity": 31, "prefix_tokens": 543,
  "tokenizer": {"sep": 235248, "neg": 728,
    "pieces": {"0": 235276, "1": 235274, "2": 235284, "3": 235304, "4": 235310,
               "5": 235308, "6": 235318, "7": 235324, "8": 235321, "9": 235315},
    "tail": [235289, 108, 4022, 235292, 235248]},
  "tasks": {"scoop sand and dump it to the left": {"head": [2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
            "move rock to container": {"head": [2, 31, 32, 33, 34, 35, 36, 37, 38, 39]}},
  "kaivuri": {"fps": 30, "state": {"names": ["lift", "tilt", "scoop"]},
              "action": {"names": ["slew", "lift", "tilt", "scoop"]}}})";
static const char *norm_yaml = R"({"norm_stats": {
  "state": {"q01": [-30.25, 10.5, -90.125], "q99": [60.75, 120.0, 45.5]},
  "actions": {"q01": [-0.9674466338157653, -0.938210240650177, -0.9261431825160981, -0.9327842418551445],
              "q99": [0.967168907570839, 0.9412212884902955, 0.9908881056308747, 0.9627036624789238]}}})";

static void prompt_tests() {
    expect(state_bin(-1.0000001) == -1 && state_bin(-1) == 0 && state_bin(0) == 128 &&
               state_bin(0.9921875) == 255 && state_bin(5) == 255 &&
               state_bin(std::nextafter(0.0, -1.0)) == 127,
           "digitize edges");
    auto contract = load_contract(YAML::Load(bundle_yaml), YAML::Load(norm_yaml));
    expect(contract.state_joints == std::vector<int>({1, 2, 3}) && contract.fps == 30 &&
               contract.robot_dim == 4 && contract.prompt.longest(3) == 31,
           "contract fields");
    for (auto &c : bin_cases) {
        auto bins = contract.bins({c.state[0], c.state[1], c.state[2]});
        expect(bins == std::vector<int>({c.bins[0], c.bins[1], c.bins[2]}),
               "numpy bins for state " + std::to_string(c.state[0]));
    }
    for (size_t i = 0; i < sizeof(radian_text) / sizeof(radian_text[0]); ++i) {
        auto r = robot_state(
            status_json(1000000000, radian_text[i][0], radian_text[i][1], radian_text[i][2]),
            contract.state_joints, 1000000000, 50000000);
        for (int j = 0; j < 3; ++j)
            expect(r.degrees[j] == bin_cases[i].state[j], "float32 rad2deg like numpy");
    }
    const auto &t = contract.prompt;
    const std::string sand = "scoop sand and dump it to the left";
    auto ids = contract.prompt.compose(sand, {-1, 0, 255});
    std::vector<int64_t> want = t.heads.at(sand);
    for (int64_t v :
         {t.neg, t.digits[1], t.sep, t.digits[0], t.sep, t.digits[2], t.digits[5], t.digits[5]})
        want.push_back(v);
    want.insert(want.end(), t.tail.begin(), t.tail.end());
    expect(ids == want && ids.size() == 27, "'-1 0 255' pieces");
    expect(contract.prompt.compose("move rock to container", {7, 42, 128}).size() == 10 + 9 + 5,
           "short task prompt");
    expect_throw([&] { contract.prompt.compose("unknown", {1, 2, 3}); }, "unknown task");
    expect_throw([&] { contract.bins({1.f, NAN, 2.f}); }, "NaN state");
    auto small = YAML::Load(bundle_yaml);
    small["prompt_capacity"] = 30;
    expect_throw([&] { load_contract(small, YAML::Load(norm_yaml)); }, "prompt capacity");
    auto order = YAML::Load(bundle_yaml);
    order["kaivuri"]["action"]["names"][0] = "lift";
    expect_throw([&] { load_contract(order, YAML::Load(norm_yaml)); }, "action order");
    for (auto &c : unnormalize_cases)
        for (int j = 0; j < 4; ++j)
            expect(unnormalize(c.x[j], contract.actions.q01[j], contract.actions.q99[j]) == c.y[j],
                   "numpy unnormalize");
    expect_throw(
        [&] {
            robot_state(status_json(1000000000, "0", "0", "0", false), {1, 2, 3}, 1000000000,
                        50000000);
        },
        "unhealthy IMU state");
    expect_throw(
        [&] {
            robot_state(status_json(1000000000, "0", "0", "0"), {1, 2, 3}, 1060000000, 50000000);
        },
        "stale IMU state");
    expect_throw(
        [&] {
            robot_state(status_json(1020000000, "0", "0", "0"), {1, 2, 3}, 1000000000, 50000000);
        },
        "future IMU state");
}

static void resize_tests() {
    std::vector<uint8_t> g(47 * 33 * 2);
    for (int y = 0; y < 33; ++y)
        for (int x = 0; x < 47; ++x) // Two interleaved channels; the second is ignored.
            g[(y * 47 + x) * 2] = uint8_t((x * 37 + y * 91 + (x * y) % 23 * 5) % 256);
    auto small = pil_bilinear(g.data(), 47, 33, 47 * 2, 2, 16, 11);
    expect(std::equal(small.begin(), small.end(), pil_16x11), "Pillow bilinear downscale");
    auto wide = pil_bilinear(g.data(), 47, 33, 47 * 2, 2, 53, 7);
    expect(std::equal(wide.begin(), wide.end(), pil_53x7), "Pillow bilinear mixed scale");
}

static void plan_tests() {
    std::vector<double> robot(50 * 4);
    for (int r = 0; r < 50; ++r) {
        robot[r * 4 + 0] = 0.5;
        robot[r * 4 + 1] = r == 3 ? 1.03 : 0.25;
        robot[r * 4 + 2] = -0.5;
        robot[r * 4 + 3] = r == 4 ? -1.01 : 0.1;
    }
    int clamped = 0;
    ActionGate gate;
    auto plan = make_plan(robot, 50, 4, 30, 5000000000ULL, 4900000000ULL, gate, clamped);
    expect(clamped == 2 && plan.rows[3][1] == 1 && plan.rows[4][3] == -1, "tolerance clamp");
    expect(plan.rows[0][0] == 0 && plan.rows[0][4] == 0 && plan.rows[0][5] == 0,
           "slew and tracks stay zero without their flags");
    gate.slew = true;
    plan = make_plan(robot, 50, 4, 30, 5000000000ULL, 4900000000ULL, gate, clamped);
    expect(plan.rows[0][0] == 0.5, "explicit slew flag passes slew");
    expect(plan.end_ns() == 5000000000ULL + 1633333333ULL, "49 rows at 30 fps");
    robot[7] = 1.06;
    auto overshoot = make_plan(robot, 50, 4, 30, 1, 1, gate, clamped);
    expect(clamped == 3 && overshoot.rows[1][3] == 1, "default clamps any overshoot");
    gate.tolerance = 0.05;
    expect_throw([&] { make_plan(robot, 50, 4, 30, 1, 1, gate, clamped); }, "beyond tolerance");
    robot[7] = NAN;
    expect_throw([&] { make_plan(robot, 50, 4, 30, 1, 1, gate, clamped); }, "nonfinite action");
    robot[7] = 0;
    gate.tolerance = ActionGate{}.tolerance;
    // Wire format: the controller's own decoder accepts the plan unchanged.
    Config cfg;
    const uint64_t now = 5300000000ULL;
    auto bytes = encode(Operation::Drive, deadman_flag, 9, now, &plan);
    Packet p;
    expect(decode_packet(bytes.data(), bytes.size(), now * 1e-9, cfg, p), "controller decodes");
    expect(p.count == 50 && p.fps == 30 && p.sequence == 9 && p.flags == deadman_flag &&
               std::abs(p.t0 - 5.0) < 1e-9 && std::abs(p.observation - 4.9) < 1e-9 &&
               p.steps[0][1] == float(0.25) && p.steps[3][1] == 1,
           "decoded plan fields");
    auto renewal = encode(Operation::Drive, deadman_flag, 10, now + 100000000ULL, &plan);
    expect(std::memcmp(bytes.data() + 28, renewal.data() + 28, bytes.size() - 28) == 0,
           "renewal keeps rows, fps, t0 and observation");
    auto disarm = encode(Operation::Disarm, 0, 11, now, nullptr);
    expect(disarm.size() == wire_header &&
               decode_packet(disarm.data(), disarm.size(), now * 1e-9, cfg, p) &&
               p.op == Operation::Disarm,
           "disarm packet");
}

static void supervisor_tests() {
    const uint64_t ms = 1000000ULL, base = 1000 * ms;
    SupervisorConfig cfg; // renew 100 ms
    cfg.max_observation_age_ns = 1000 * ms;
    PlanSupervisor s(cfg);
    std::string why;
    s.hold(base);
    expect(s.next(base, why) == PlanSupervisor::Decision::Send && s.fresh(), "hold sent first");
    s.sent(base);
    expect(s.next(base + 50 * ms, why) == PlanSupervisor::Decision::Idle, "hold within renewal");
    expect(s.next(base + 100 * ms, why) == PlanSupervisor::Decision::Send, "hold renewal");
    s.sent(base + 100 * ms);
    Plan p{std::vector<Commands>(50), 30, base + 200 * ms, base + 200 * ms};
    expect(s.offer(p, base + 650 * ms).empty(), "first chunk accepted");
    expect(s.next(base + 651 * ms, why) == PlanSupervisor::Decision::Send && s.fresh(),
           "new chunk sent at once");
    s.sent(base + 651 * ms);
    expect(!s.offer(p, base + 700 * ms).empty(), "same observation refused");
    Plan older = p;
    older.observation_ns -= 10 * ms;
    older.t0_ns -= 10 * ms;
    expect(!s.offer(older, base + 700 * ms).empty(), "older observation refused");
    expect(s.next(base + 751 * ms, why) == PlanSupervisor::Decision::Send && !s.fresh() &&
               s.plan().observation_ns == base + 200 * ms && s.plan().t0_ns == base + 200 * ms,
           "renewal keeps observation and t0");
    s.sent(base + 751 * ms);
    expect(s.next(base + 1199 * ms, why) == PlanSupervisor::Decision::Send, "renewals continue");
    s.sent(base + 1199 * ms);
    expect(s.next(base + 1201 * ms, why) == PlanSupervisor::Decision::Stop &&
               why.find("observation age") != std::string::npos,
           "stalled inference stops at the observation age limit");
    expect(!s.offer(Plan{std::vector<Commands>(50), 30, base + 1150 * ms, base + 1150 * ms},
                    base + 1210 * ms)
                .empty(),
           "stopped supervisor refuses new plans");
    PlanSupervisor e(cfg);
    e.hold(base);
    Plan short_plan{std::vector<Commands>(10), 30, base, base}; // 300 ms of rows
    expect(e.offer(short_plan, base + 100 * ms).empty(), "short chunk accepted");
    expect(e.next(base + 301 * ms, why) == PlanSupervisor::Decision::Stop &&
               why.find("exhausted") != std::string::npos,
           "exhausted chunk stops");
    PlanSupervisor late(cfg);
    late.hold(base);
    expect(!late.offer(Plan{std::vector<Commands>(50), 30, base, base}, base + 1001 * ms).empty(),
           "chunk older than the age limit refused");
    expect(!late.offer(Plan{std::vector<Commands>(50), 30, base + 50 * ms, base}, base).empty(),
           "future plan start refused");
    expect(late.next(base + 5001 * ms, why) == PlanSupervisor::Decision::Stop,
           "first-chunk timeout");
}

struct Daemon {
    pid_t pid = -1;
    std::string socket;
    Daemon(const std::string &binary, const std::string &dir, bool slew) {
        socket = dir + "/control.sock";
        fs::remove(socket);
        pid = fork();
        if (pid == 0) {
            std::string profile =
                std::string(NATIVE_SOURCE) + "/../configuration_files/profiles/jetson";
            std::string safety = std::string(NATIVE_SOURCE) + "/config/safety.yaml";
            std::vector<const char *> args{binary.c_str(), "--profile",     profile.c_str(),
                                           "--safety",     safety.c_str(),  "--socket",
                                           socket.c_str(), "--simulate-imu"};
            if (slew)
                args.push_back("--enable-slew");
            args.push_back(nullptr);
            execv(binary.c_str(), const_cast<char *const *>(args.data()));
            _exit(127);
        }
        for (int i = 0; i < 300 && !fs::exists(socket); ++i)
            sleep_ms(10);
        expect(fs::exists(socket), "control socket");
    }
    ~Daemon() {
        kill(pid, SIGTERM);
        int status = 0;
        waitpid(pid, &status, 0);
    }
};
static YAML::Node status(ControlLink &link) {
    return YAML::Load(link.status());
}
static Plan chunk(uint64_t observation, int rows, double boom) {
    Plan p{std::vector<Commands>(rows, Commands{0.4, boom, -0.1, 0.05, 0, 0}), 30, observation,
           observation};
    return p;
}
static void producer_tests(const std::string &binary) {
    const std::string dir =
        fs::temp_directory_path() / ("kaivuri-pi05-test-" + std::to_string(getpid()));
    fs::create_directories(dir);
    {
        Daemon daemon(binary, dir, false);
        ControlLink watch(daemon.socket);
        auto robot = robot_state(watch.status(), {1, 2, 3}, now_ns(), 50000000);
        expect(robot.degrees.size() == 3 && robot.imu_ns > 0, "simulated status gives state");
        ProducerOptions options;
        options.socket = daemon.socket;
        options.gate.slew = true;
        expect_throw([&] { Producer p(options); }, "slew without the daemon flag");
        options.gate.slew = false;
        options.supervisor.max_observation_age_ns = 600000000ULL;
        std::string reason;
        {
            Producer producer(options);
            producer.arm();
            expect(status(watch)["state"].as<std::string>() == "armed", "armed by producer");
            sleep_ms(650); // centered hold renewed past the 500 ms lease
            expect(status(watch)["state"].as<std::string>() == "armed", "hold renewed");
            const uint64_t observed = now_ns();
            expect(producer.post(chunk(observed, 50, 0.2), reason), "chunk posted: " + reason);
            sleep_ms(60);
            auto s = status(watch);
            auto applied = s["applied_action"];
            expect(s["state"].as<std::string>() == "armed" &&
                       std::abs(applied[1].as<double>() - 0.2) < 1e-6 &&
                       applied[0].as<double>() == 0,
                   "plan applied; slew gated");
            expect(!producer.post(chunk(observed, 50, 0.3), reason), "same observation refused");
            sleep_ms(400);
            expect(status(watch)["state"].as<std::string>() == "armed", "renewed past lease");
            sleep_ms(300); // 760 ms > 600 ms observation limit
            expect(!producer.running() &&
                       producer.stop_reason().find("observation age") != std::string::npos,
                   "stops at the observation age limit: " + producer.stop_reason());
            auto c = producer.counters();
            expect(c.plans == 1 && c.renewals >= 4 && c.holds >= 5 && c.max_send_gap_ms < 150,
                   "producer counters");
            s = status(watch);
            expect(s["state"].as<std::string>() == "disarmed" &&
                       s["reason"].as<std::string>() == "explicit_disarm",
                   "producer disarmed on stop");
        }
        {
            Producer producer(options);
            producer.arm();
            expect(producer.post(chunk(now_ns(), 10, 0.1), reason), "short chunk posted");
            sleep_ms(400);
            expect(!producer.running() &&
                       producer.stop_reason().find("exhausted") != std::string::npos,
                   "stops when the chunk is exhausted");
            expect(status(watch)["state"].as<std::string>() == "disarmed", "disarmed after");
        }
        {
            Producer producer(options);
            producer.arm();
            expect(producer.post(chunk(now_ns(), 50, 0.1), reason), "chunk posted");
            watch.send(Operation::Estop);
            sleep_ms(100);
            expect(!producer.running() &&
                       producer.stop_reason().find("emergency_stop") != std::string::npos,
                   "stops on controller fault: " + producer.stop_reason());
            expect(!producer.post(chunk(now_ns(), 50, 0.1), reason), "no posts after stop");
        }
        expect(status(watch)["state"].as<std::string>() == "disarmed",
               "fault cleared only by the producer's own disarm");
        watch.send(Operation::Estop);
        sleep_ms(30);
        expect_throw([&] { Producer p(options); }, "refuses a faulted controller");
        watch.send(Operation::Disarm);
        sleep_ms(30);
        {
            Producer producer(options);
            producer.arm();
        } // Destructor disarms and releases ownership without a disconnect fault.
        sleep_ms(30);
        auto s = status(watch);
        expect(s["state"].as<std::string>() == "disarmed" &&
                   s["reason"].as<std::string>() == "explicit_disarm",
               "clean producer shutdown");
    }
    {
        Daemon daemon(binary, dir, true);
        ProducerOptions options;
        options.socket = daemon.socket;
        options.gate.slew = true;
        Producer producer(options);
        producer.arm();
        std::string reason;
        expect(producer.post(chunk(now_ns(), 50, 0), reason), "slew chunk posted");
        sleep_ms(60);
        ControlLink watch(daemon.socket);
        expect(std::abs(status(watch)["applied_action"][0].as<double>() - 0.4) < 1e-6,
               "slew passes with both flags");
    }
    fs::remove_all(dir);
}

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("kaivuri-control path required");
        prompt_tests();
        resize_tests();
        plan_tests();
        supervisor_tests();
        producer_tests(argv[1]);
        std::cout << "pi05 tests passed: " << checks << " checks\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
