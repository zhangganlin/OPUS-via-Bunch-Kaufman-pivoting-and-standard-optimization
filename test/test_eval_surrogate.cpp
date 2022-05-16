#include "surrogate.hpp"
#include <iostream>
#include "tsc_x86.h"

using namespace std;

double evaluate_surrogate_gt( double* x, double* points,  double* lambda_c, int N, int d){
    // total flops: 3Nd + 5N + 2d + 1
    double phi, error, res = 0;
    
    // flops: 3Nd + 5N
    for(int i = 0; i < N; i++){
        phi = 0;
        // flops: 3d
        for(int j = 0; j < d; j++){
            error = x[j] - points[i * d + j];
            phi += error * error;
        }
        phi = sqrt(phi);            // flops: 1
        phi = phi * phi * phi;      // flops: 2
        // optimize: phi = sqrt(phi) * phi;
        res += phi * lambda_c[i];   // flops: 2
    }
    // flops: 2d
    for(int i = 0; i < d; i++){
        res += x[i] * lambda_c[N + i];
    }
    // flops: 1
    res += lambda_c[N + d];
    return res;
}

void test_eval1(){
    myInt64 gt_start,cur_start,gt_time,cur_time;
    int N = 154;
    int d = 4;
    int repeat = 1000000;
    int warmup_iter = 100000;
    double* x = (double*)malloc(d*sizeof(double));
    double* points = (double*)malloc(N*d*sizeof(double));
    double* lambda_c = (double*)malloc((N+d+1)*sizeof(double));
    x[0]=0.824021; x[1]=0.093371; x[2]=0.348314; x[3]=-0.233876; 
    points[0]=-0.438857; points[1]=0.585143; points[2]=0.000000; points[3]=0.731429; points[4]=-0.292571; points[5]=-0.438857; points[6]=1.024000; points[7]=0.146286; points[8]=0.438857; points[9]=0.438857; points[10]=0.292571; points[11]=1.462857; points[12]=1.170286; points[13]=0.877714; points[14]=-0.292571; points[15]=-1.462857; points[16]=-0.146286; points[17]=0.000000; points[18]=0.146286; points[19]=2.048000; 
    points[20]=-1.316571; points[21]=0.292571; points[22]=-0.731429; points[23]=-0.731429; points[24]=0.877714; points[25]=-0.146286; points[26]=1.462857; points[27]=0.877714; points[28]=-0.877714; points[29]=0.146286; points[30]=-1.609143; points[31]=1.024000; points[32]=-0.585143; points[33]=-0.292571; points[34]=1.609143; points[35]=0.585143; points[36]=0.146286; points[37]=1.316571; points[38]=0.731429; points[39]=-1.901714; 
    points[40]=-1.462857; points[41]=-0.731429; points[42]=1.316571; points[43]=1.755429; points[44]=0.292571; points[45]=1.462857; points[46]=-0.585143; points[47]=-0.438857; points[48]=0.731429; points[49]=-1.462857; points[50]=-0.438857; points[51]=-0.292571; points[52]=1.462857; points[53]=-0.877714; points[54]=0.438857; points[55]=-1.316571; points[56]=-0.588068; points[57]=0.482228; points[58]=-0.331612; points[59]=0.043024;
    points[60]=-0.618553; points[61]=1.186587; points[62]=-0.301119; points[63]=0.061555; points[64]=-0.683625; points[65]=-0.190390; points[66]=-0.244924; points[67]=0.534373; points[68]=-2.048000; points[69]=0.881324; points[70]=0.214122; points[71]=0.407836; points[72]=0.127252; points[73]=-0.191652; points[74]=-0.603791; points[75]=-0.334184; points[76]=0.573446; points[77]=-0.470070; points[78]=-0.174816; points[79]=0.211367; 
    points[80]=-0.834940; points[81]=0.250036; points[82]=0.118902; points[83]=0.524890; points[84]=-0.440669; points[85]=0.998103; points[86]=0.163663; points[87]=-0.370378; points[88]=-0.441395; points[89]=0.445805; points[90]=1.625760; points[91]=-0.025227; points[92]=-1.419071; points[93]=0.408430; points[94]=-0.878577; points[95]=1.600686; points[96]=0.941294; points[97]=1.310820; points[98]=-0.687794; points[99]=-0.520177; 
    points[100]=-0.783977; points[101]=0.759817; points[102]=-0.001204; points[103]=-0.295530; points[104]=-0.330339; points[105]=1.937459; points[106]=-0.398930; points[107]=0.119152; points[108]=-1.118988; points[109]=0.151549; points[110]=-0.526506; points[111]=-0.267048; points[112]=-0.696962; points[113]=0.407122; points[114]=-0.573621; points[115]=-0.459370; points[116]=-0.653264; points[117]=0.900259; points[118]=-0.915678; points[119]=0.044532; 
    points[120]=-1.424594; points[121]=0.234116; points[122]=-0.707761; points[123]=-0.573083; points[124]=-0.386835; points[125]=0.466866; points[126]=-0.220449; points[127]=-0.392175; points[128]=-0.212031; points[129]=0.455962; points[130]=-0.946618; points[131]=-2.016173; points[132]=0.706179; points[133]=-0.987452; points[134]=0.123982; points[135]=0.729632; points[136]=-1.739461; points[137]=0.637983; points[138]=-1.448482; points[139]=-0.081506; 
    points[140]=-0.301394; points[141]=1.291606; points[142]=0.816935; points[143]=-1.353845; points[144]=-0.649147; points[145]=0.898145; points[146]=-0.573981; points[147]=-0.290029; points[148]=-1.397388; points[149]=-0.183996; points[150]=-1.544973; points[151]=1.880812; points[152]=0.662636; points[153]=1.738856; points[154]=-2.048000; points[155]=-1.465033; points[156]=-1.397645; points[157]=0.057732; points[158]=-0.052953; points[159]=0.182227; 
    points[160]=-0.199052; points[161]=-1.673464; points[162]=-0.329263; points[163]=0.148875; points[164]=-2.048000; points[165]=0.932925; points[166]=-1.180065; points[167]=0.529598; points[168]=-0.580578; points[169]=0.421993; points[170]=-0.465839; points[171]=0.455830; points[172]=-0.160321; points[173]=0.086257; points[174]=0.496633; points[175]=0.100489; points[176]=-0.155859; points[177]=0.328610; points[178]=-0.646211; points[179]=0.724416; 
    points[180]=0.987735; points[181]=0.059787; points[182]=0.153217; points[183]=-0.621983; points[184]=-0.389782; points[185]=-0.372855; points[186]=0.495190; points[187]=-1.102465; points[188]=0.601975; points[189]=-0.125418; points[190]=0.369121; points[191]=0.525821; points[192]=0.028096; points[193]=0.576642; points[194]=0.144185; points[195]=0.172123; points[196]=-0.326380; points[197]=0.494349; points[198]=0.046410; points[199]=-0.925824; 
    points[200]=-0.076117; points[201]=0.020626; points[202]=-1.081084; points[203]=0.072713; points[204]=-0.404518; points[205]=-0.525683; points[206]=1.169155; points[207]=0.899968; points[208]=0.016797; points[209]=0.197197; points[210]=1.262630; points[211]=-0.443386; points[212]=0.553640; points[213]=0.443192; points[214]=0.155090; points[215]=0.442831; points[216]=1.045474; points[217]=-1.559576; points[218]=0.332999; points[219]=-0.131314; 
    points[220]=-0.407489; points[221]=0.173098; points[222]=-0.613137; points[223]=0.559747; points[224]=-0.135132; points[225]=0.486151; points[226]=0.248633; points[227]=0.723441; points[228]=1.027832; points[229]=-0.020093; points[230]=1.373878; points[231]=0.626023; points[232]=1.031252; points[233]=0.051391; points[234]=0.664871; points[235]=1.503614; points[236]=-0.023774; points[237]=0.121145; points[238]=-0.118618; points[239]=1.020700; 
    points[240]=0.773586; points[241]=-0.015822; points[242]=-0.057078; points[243]=1.288359; points[244]=0.512611; points[245]=0.788251; points[246]=0.265494; points[247]=0.319292; points[248]=1.357769; points[249]=0.461158; points[250]=1.317282; points[251]=0.700844; points[252]=-0.018472; points[253]=-0.109874; points[254]=-0.384251; points[255]=1.375397; points[256]=0.746780; points[257]=-0.391937; points[258]=-0.494895; points[259]=0.706174; 
    points[260]=1.739487; points[261]=0.069445; points[262]=1.860512; points[263]=0.176470; points[264]=-0.408689; points[265]=-0.800531; points[266]=2.048000; points[267]=1.350384; points[268]=1.977655; points[269]=0.724495; points[270]=0.306917; points[271]=0.633015; points[272]=1.259068; points[273]=-0.834860; points[274]=-0.202713; points[275]=-0.019165; points[276]=1.141722; points[277]=-0.140067; points[278]=0.598831; points[279]=0.512550; 
    points[280]=0.428512; points[281]=0.473466; points[282]=0.480623; points[283]=0.464920; points[284]=0.749315; points[285]=0.421117; points[286]=0.967360; points[287]=0.763530; points[288]=0.038573; points[289]=-0.064537; points[290]=0.457097; points[291]=0.943421; points[292]=0.060995; points[293]=0.845112; points[294]=-0.135385; points[295]=0.198834; points[296]=1.129164; points[297]=0.146118; points[298]=-0.484232; points[299]=0.774940; 
    points[300]=0.465257; points[301]=0.983916; points[302]=0.077631; points[303]=0.194824; points[304]=0.512258; points[305]=0.363382; points[306]=0.575190; points[307]=0.784830; points[308]=0.681048; points[309]=0.693976; points[310]=0.084932; points[311]=-0.920827; points[312]=1.227411; points[313]=0.337919; points[314]=0.534173; points[315]=0.822412; points[316]=-0.685712; points[317]=0.344158; points[318]=0.345275; points[319]=0.363694; 
    points[320]=-0.329495; points[321]=0.465339; points[322]=0.507782; points[323]=1.981006; points[324]=0.634515; points[325]=0.383358; points[326]=0.289784; points[327]=0.459221; points[328]=0.076525; points[329]=0.563650; points[330]=-0.362747; points[331]=0.093640; points[332]=-0.882740; points[333]=0.120852; points[334]=1.129794; points[335]=0.525875; points[336]=-0.602262; points[337]=0.365901; points[338]=0.461488; points[339]=0.273906; 
    points[340]=-0.689785; points[341]=0.574285; points[342]=0.339823; points[343]=-0.037465; points[344]=-0.923835; points[345]=0.318212; points[346]=0.136041; points[347]=-0.045775; points[348]=-1.198958; points[349]=0.745038; points[350]=-0.084921; points[351]=-1.110357; points[352]=0.583186; points[353]=0.374534; points[354]=0.107082; points[355]=-1.655603; points[356]=-0.388641; points[357]=0.409659; points[358]=0.105097; points[359]=0.153907; 
    points[360]=-0.955526; points[361]=0.660473; points[362]=-0.768929; points[363]=-0.870715; points[364]=-0.243443; points[365]=0.931503; points[366]=0.447759; points[367]=-2.048000; points[368]=-0.113607; points[369]=0.181178; points[370]=0.412504; points[371]=0.049344; points[372]=-2.048000; points[373]=0.764721; points[374]=-0.765745; points[375]=0.421764; points[376]=-0.567871; points[377]=1.422366; points[378]=-0.739135; points[379]=-0.292289; 
    points[380]=-0.742204; points[381]=0.260363; points[382]=0.287792; points[383]=0.008377; points[384]=-0.814197; points[385]=1.586619; points[386]=-0.445610; points[387]=0.140932; points[388]=-1.812866; points[389]=0.704469; points[390]=0.314693; points[391]=0.054046; points[392]=-1.358183; points[393]=0.314206; points[394]=0.402962; points[395]=0.060686; points[396]=-1.739990; points[397]=0.686061; points[398]=-0.118131; points[399]=-0.622003; 
    points[400]=-1.321694; points[401]=0.695755; points[402]=0.102268; points[403]=-0.761595; points[404]=-0.904991; points[405]=0.271689; points[406]=-0.062519; points[407]=-0.753003; points[408]=-1.298974; points[409]=-0.050001; points[410]=0.191854; points[411]=-1.044141; points[412]=-1.138998; points[413]=0.137799; points[414]=0.127611; points[415]=0.058625; points[416]=-0.942143; points[417]=0.776673; points[418]=0.152115; points[419]=-0.931782; 
    points[420]=-1.033944; points[421]=0.315492; points[422]=0.418679; points[423]=-0.035462; points[424]=-1.166262; points[425]=0.081116; points[426]=0.216062; points[427]=-0.540755; points[428]=-1.032508; points[429]=0.303377; points[430]=-0.505692; points[431]=0.073790; points[432]=-0.669031; points[433]=0.071515; points[434]=-0.785761; points[435]=-1.625935; points[436]=-0.627504; points[437]=0.322367; points[438]=0.333426; points[439]=0.123897; 
    points[440]=-0.250450; points[441]=0.538701; points[442]=0.162808; points[443]=0.029476; points[444]=-0.342817; points[445]=0.628189; points[446]=-0.423140; points[447]=0.196451; points[448]=-0.850233; points[449]=0.350220; points[450]=0.290283; points[451]=0.098293; points[452]=-0.829918; points[453]=0.595374; points[454]=0.245023; points[455]=-0.029901; points[456]=-0.592039; points[457]=0.567384; points[458]=0.395787; points[459]=0.374239; 
    points[460]=-0.049049; points[461]=0.334655; points[462]=0.297770; points[463]=-0.189445; points[464]=-0.563957; points[465]=0.099571; points[466]=-0.131568; points[467]=1.836932; points[468]=-0.824974; points[469]=0.591303; points[470]=0.304940; points[471]=-0.021216; points[472]=0.189279; points[473]=0.389605; points[474]=1.009209; points[475]=0.905734; points[476]=-1.338501; points[477]=-0.124568; points[478]=0.392424; points[479]=1.449654; 
    points[480]=-0.432340; points[481]=0.517697; points[482]=0.187566; points[483]=-0.268258; points[484]=0.103592; points[485]=0.328059; points[486]=0.240676; points[487]=-0.074343; points[488]=-0.901997; points[489]=0.157534; points[490]=-0.101590; points[491]=0.117836; points[492]=-0.642333; points[493]=0.447664; points[494]=0.331722; points[495]=-0.001608; points[496]=-0.167661; points[497]=-0.214215; points[498]=0.671717; points[499]=-0.119942; 
    points[500]=0.497681; points[501]=0.044969; points[502]=-0.084777; points[503]=0.093251; points[504]=-0.361605; points[505]=0.390792; points[506]=0.284327; points[507]=0.334842; points[508]=0.016976; points[509]=0.507332; points[510]=0.554766; points[511]=0.396315; points[512]=-0.402096; points[513]=0.481335; points[514]=0.500982; points[515]=0.720317; points[516]=0.574611; points[517]=0.391850; points[518]=0.563580; points[519]=0.349359; 
    points[520]=0.132733; points[521]=0.259337; points[522]=-0.131983; points[523]=0.211097; points[524]=-0.595814; points[525]=0.922249; points[526]=0.434346; points[527]=-0.079481; points[528]=0.486078; points[529]=0.379977; points[530]=1.325821; points[531]=1.375398; points[532]=-0.968595; points[533]=0.632804; points[534]=0.327960; points[535]=0.225768; points[536]=-0.372893; points[537]=0.464728; points[538]=0.338310; points[539]=0.180527; 
    points[540]=-0.380562; points[541]=0.357291; points[542]=0.842477; points[543]=-0.154327; points[544]=-1.008876; points[545]=0.245749; points[546]=0.436388; points[547]=1.319685; points[548]=-0.776876; points[549]=0.574207; points[550]=0.294402; points[551]=-0.096345; points[552]=-0.256883; points[553]=0.328695; points[554]=0.250837; points[555]=-0.004867; points[556]=0.371996; points[557]=0.212572; points[558]=0.231795; points[559]=-0.021635; 
    points[560]=0.304385; points[561]=0.242052; points[562]=0.340175; points[563]=0.155978; points[564]=0.650022; points[565]=0.306638; points[566]=0.214373; points[567]=0.281479; points[568]=0.404218; points[569]=0.469312; points[570]=0.074110; points[571]=0.083117; points[572]=0.808670; points[573]=0.352706; points[574]=0.341462; points[575]=0.501816; points[576]=0.996951; points[577]=0.320377; points[578]=0.259425; points[579]=-1.312737; 
    points[580]=-0.289769; points[581]=0.096420; points[582]=0.307280; points[583]=-0.037740; points[584]=0.665237; points[585]=0.485064; points[586]=0.158806; points[587]=0.353942; points[588]=0.272641; points[589]=1.010267; points[590]=0.268376; points[591]=-0.840731; points[592]=0.394026; points[593]=0.305912; points[594]=0.308300; points[595]=0.285718; points[596]=0.364540; points[597]=0.238604; points[598]=0.370942; points[599]=0.036293; 
    points[600]=0.415775; points[601]=0.279892; points[602]=0.245917; points[603]=-0.033224; points[604]=-0.708464; points[605]=0.396655; points[606]=0.269926; points[607]=0.020304; points[608]=0.430346; points[609]=0.637833; points[610]=-0.066549; points[611]=0.060422; points[612]=0.280278; points[613]=0.334881; points[614]=0.462814; points[615]=-0.105473; 

    lambda_c[0]=-1025.645508; lambda_c[1]=-574.705568; lambda_c[2]=-908.740760; lambda_c[3]=-576.505841; lambda_c[4]=-951.937705; lambda_c[5]=-11263.355156; lambda_c[6]=-3025.367756; lambda_c[7]=-454.580106; lambda_c[8]=-1340.011660; lambda_c[9]=-947.615895; lambda_c[10]=-561.823049; lambda_c[11]=-2904.099865; lambda_c[12]=-983.810073; lambda_c[13]=-364.250925; lambda_c[14]=-152.122880; lambda_c[15]=788.450247; lambda_c[16]=-697.442159; lambda_c[17]=392.330823; lambda_c[18]=167.068098; lambda_c[19]=-113.842606; lambda_c[20]=680.459808;
    lambda_c[21]=652.199728; lambda_c[22]=699.679125; lambda_c[23]=-206.777284; lambda_c[24]=1009.053225; lambda_c[25]=-20.713258; lambda_c[26]=1584.613279; lambda_c[27]=-1370.199337; lambda_c[28]=977.465268; lambda_c[29]=-51.280421; lambda_c[30]=9871.923847; lambda_c[31]=-176.388059; lambda_c[32]=-314.308292; lambda_c[33]=-571.579399; lambda_c[34]=-98.009103; lambda_c[35]=344.352019; lambda_c[36]=-760.932578; lambda_c[37]=162.940987; lambda_c[38]=297.228822; lambda_c[39]=-644.491872; lambda_c[40]=544.979668;
    lambda_c[41]=-17.843522; lambda_c[42]=814.998334; lambda_c[43]=218.826272; lambda_c[44]=-15.623110; lambda_c[45]=-379.658263; lambda_c[46]=-249.359368; lambda_c[47]=144.786500; lambda_c[48]=-245.775520; lambda_c[49]=-675.317508; lambda_c[50]=270.227831; lambda_c[51]=727.538003; lambda_c[52]=-76.240222; lambda_c[53]=-31.768185; lambda_c[54]=473.126679; lambda_c[55]=62.241214; lambda_c[56]=848.339683; lambda_c[57]=3013.849309; lambda_c[58]=662.413470; lambda_c[59]=156.987146; lambda_c[60]=607.634365; 
    lambda_c[61]=-742.204953; lambda_c[62]=-1201.164857; lambda_c[63]=179.018284; lambda_c[64]=96.347366; lambda_c[65]=403.826245; lambda_c[66]=638.303057; lambda_c[67]=101.421301; lambda_c[68]=915.358991; lambda_c[69]=-645.160987; lambda_c[70]=200.839695; lambda_c[71]=102.219796; lambda_c[72]=110.103815; lambda_c[73]=513.868703; lambda_c[74]=-637.491339; lambda_c[75]=293.897136; lambda_c[76]=314.634325; lambda_c[77]=-111.437536; lambda_c[78]=122.674141; lambda_c[79]=-123.790305; lambda_c[80]=498.251989; 
    lambda_c[81]=183.014475; lambda_c[82]=208.478983; lambda_c[83]=704.181325; lambda_c[84]=-559.175792; lambda_c[85]=990.375408; lambda_c[86]=-121.936180; lambda_c[87]=-10.883107; lambda_c[88]=361.933460; lambda_c[89]=-287.430658; lambda_c[90]=2381.169324; lambda_c[91]=673.950129; lambda_c[92]=50.146750; lambda_c[93]=6.189162; lambda_c[94]=874.099443; lambda_c[95]=156.242305; lambda_c[96]=-1711.026844; lambda_c[97]=-413.319045; lambda_c[98]=359.476591; lambda_c[99]=-135.032131; lambda_c[100]=200.394329; 
    lambda_c[101]=1263.164238; lambda_c[102]=231.210636; lambda_c[103]=103.993834; lambda_c[104]=-808.182315; lambda_c[105]=-174.627124; lambda_c[106]=-381.932258; lambda_c[107]=-392.547599; lambda_c[108]=596.460628; lambda_c[109]=286.704798; lambda_c[110]=3.014119; lambda_c[111]=-184.183226; lambda_c[112]=34.670911; lambda_c[113]=1059.140625; lambda_c[114]=518.003251; lambda_c[115]=-88.676225; lambda_c[116]=801.759385; lambda_c[117]=-279.432935; lambda_c[118]=59.926173; lambda_c[119]=766.838623; lambda_c[120]=36.410149; 
    lambda_c[121]=443.253554; lambda_c[122]=449.913754; lambda_c[123]=108.616476; lambda_c[124]=632.558319; lambda_c[125]=-30.582044; lambda_c[126]=117.157762; lambda_c[127]=-365.477117; lambda_c[128]=-292.773520; lambda_c[129]=-456.420495; lambda_c[130]=-589.350284; lambda_c[131]=-1282.564050; lambda_c[132]=441.461004; lambda_c[133]=-516.541504; lambda_c[134]=308.574260; lambda_c[135]=-345.638574; lambda_c[136]=-884.145288; lambda_c[137]=-436.773041; lambda_c[138]=159.798251; lambda_c[139]=-303.669522; lambda_c[140]=164.930906; 
    lambda_c[141]=60.747162; lambda_c[142]=-167.563233; lambda_c[143]=85.086316; lambda_c[144]=563.691916; lambda_c[145]=-368.385991; lambda_c[146]=23.226543; lambda_c[147]=902.981998; lambda_c[148]=98.612159; lambda_c[149]=297.449193; lambda_c[150]=226.902049; lambda_c[151]=-723.982869; lambda_c[152]=567.000566; lambda_c[153]=-701.764077; lambda_c[154]=301.144511; lambda_c[155]=1272.400533; lambda_c[156]=-832.985018; lambda_c[157]=98.960879; lambda_c[158]=-5564.671793; 

    double groundtruth,result;

    for(int i = 0; i < warmup_iter; i++){
        groundtruth = evaluate_surrogate_gt(x,points,lambda_c,N,d);
    }
    gt_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        groundtruth = evaluate_surrogate_gt(x,points,lambda_c,N,d);
    }
    gt_time = stop_tsc(gt_start);
    cout << "groundtruth: "<<groundtruth<<endl; 



    for(int i = 0; i < warmup_iter; i++){
        groundtruth = evaluate_surrogate(x,points,lambda_c,N,d);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        result = evaluate_surrogate(x,points,lambda_c,N,d);
    }
    cur_time = stop_tsc(cur_start);
    cout << "current result: "<< result << endl;


    cout << "groundtruth cycles: "<< gt_time/(double)repeat << endl;
    cout << "current cycles: "<< cur_time/(double)repeat << endl;

}


int main(){
    test_eval1();
}