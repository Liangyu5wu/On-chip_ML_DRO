#include <TROOT.h>
#include <TChain.h>
#include <TChainElement.h>
#include <TFile.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TH1.h>
#include <TH2.h>
#include <TF1.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <TProfile2D.h>
#include <TCanvas.h>
#include <TMultiGraph.h>
#include <Riostream.h>
#include <TVirtualFFT.h>
#include <TLegend.h>
#include <H5Cpp.h>

constexpr unsigned int N_Channels = 2 * 10;
constexpr unsigned int N_Samples = 1024;

TGraph *grCerenkov;
TGraph *grScint;

pair<double,double> baseline(TGraph*, double, double);
double funcSDL(double*, double*);
TGraph *grGetScintillation(TGraph*, double, double, double, double, double);







class TreeReader {
public:
  TreeReader(TString filename, TString treename="tree") {
    tree_ = new TChain(treename);
    tree_->Add(filename);

    /*
    std::cout << "The following files were added to the TChain:\n";
    const auto& fileElements = *tree_->GetListOfFiles();
    for (TObject const* op : fileElements) {
      auto chainElement = static_cast<const TChainElement*>(op);
      std::cout << chainElement->GetTitle() << "\n";
    }
    
    tree_->Print();
    */
    tree_->SetBranchAddress("horizontal_interval", &horizontal_interval_);
    tree_->SetBranchAddress("channels", channels_[0].data());
    tree_->SetBranchAddress("trigger", trigger_[0].data());

  }
  
  UInt_t num_entries() const {
    //return 5000; // std::min(5000, tree_->GetEntries());
    return tree_->GetEntries();
  }

  void get_entry(UInt_t i) {
    tree_->GetEntry(i);
  }

  const std::array<Double_t, N_Samples> time() const {
    std::array<Double_t, N_Samples> t;
    for (unsigned int i = 0; i < N_Samples; i++) {
      t[i] = static_cast<Double_t>(i) * horizontal_interval_;
    }
    return t;
  }

  std::array<Double_t, N_Samples> voltages(UInt_t channel) const {
    std::array<Double_t, N_Samples> volts;
    for (unsigned int i = 0; i < N_Samples; i++) {
      volts[i] = static_cast<Double_t>(channels_[channel][i]);
    }
    return volts;
  }

  std::array<Double_t, N_Samples> trigger(UInt_t channel) const {
    std::array<Double_t, N_Samples> volts;
    for (unsigned int i = 0; i < N_Samples; i++) {
      volts[i] = static_cast<Double_t>(trigger_[channel][i]);
    }
    return volts;
  }

  
  Float_t horizontal_interval() const { return horizontal_interval_; }

  UInt_t get_bin(Double_t t) const {
    return static_cast<UInt_t>(t / horizontal_interval_);
  }

private:
  TChain* tree_;

  Float_t horizontal_interval_;
  std::array<std::array<Float_t, N_Samples>, N_Channels> channels_;
  std::array<std::array<Float_t, N_Samples>, 2> trigger_;
  std::array<Double_t, N_Samples> times_;
};



class RecoTRG{
public:
    RecoTRG(TGraph *gr){

      pair<double,double> p = baseline(gr,   0, 100);
      pair<double,double> a = baseline(gr, 400, 1000);
      ped_ = p.first;
      rms_ = p.second;
      amp_ = a.first - p.first;

      int     N = gr->GetN();
      double *X = gr->GetX();
      double *Y = gr->GetY();
      // Timing of the threshold using linear interpolation
      double threshold = ped_ + 0.5 * amp_;
      time_ = 0;
      for(int i=1; i<N; i++){
          if( Y[i-1] <= threshold && Y[i] > threshold){
              time_ = X[i-1] + (threshold - Y[i-1])/(Y[i] - Y[i-1]) * (X[i] - X[i-1]);
              break;
          }
      }
    }
            
    double ped()  { return ped_; }
    double rms()  { return rms_; }
    double amp()  { return amp_; }
    double time() { return time_; }
        

private:
    double ped_;
    double rms_;
    double amp_;
    double time_;
};



pair<double,double> baseline(TGraph *gr, double tmin=0, double tmax=1000.)
{
  int     N = gr->GetN();
  double *X = gr->GetX();
  double *Y = gr->GetY();
  
  std::vector<double> v;
  for(int i=0; i<N; i++){
    if(X[i] > tmin && X[i] < tmax)
      v.push_back(Y[i]);
  }
  
  double vmin = 0.;
  double vmax = 0.;
  std::sort(v.begin(),v.end());
  unsigned int nInInterval = int(0.68*v.size());
  double minInterval = 1e+9;
  for (unsigned int i=0; i<v.size(); i++) {
    double interval_size=minInterval;
    if ((i+nInInterval)<v.size()) interval_size = (v[i+nInInterval]-v[i]);
    if (interval_size < minInterval){
      minInterval = interval_size;
      vmin = v[i];
      vmax = vmin + minInterval;
    }
  }
  
  pair<double,double> result;
  result.first = 0.5 * (vmax + vmin);
  result.second = 0.5 * (vmax - vmin);
  return result;
}




double funcSDL(double *x, double *par)
{
    double t = x[0] - par[0];
    double f = 0;
    if(t > -70){
        f += par[1] * grCerenkov->Eval(t) + par[2] * grScint->Eval(t);
    }
    return f;
}



TGraph *grGetScintillation(TGraph *gr0, double tauS=300, double tauM=300, double tauL=300, double fracS=0.0, double fracM=0.0)
{
    /*
     * Build average 1pe pulse shape for scintillation photo-electron based on average 1pe pulse shape for Cerenkov one
     * Assume that scintillation has three decay components: short, medium and long with decay times tauS, tauM and tauL, respectively
     * Fractions of photo-electrons with tauS and tauM are fracS and fracM, respectively
     */
    
    int     N = gr0->GetN();
    double *X = gr0->GetX();
    double *Y = gr0->GetY();
    
    TF1 *fDecay = new TF1("fDecay","[3]*exp(-x/[0])/[0]+[4]*exp(-x/[1])/[1]+(1-[3]-[4])*exp(-x/[2])/[2]",0,2000);
    fDecay->SetParameters(tauS,tauM,tauL,fracS,fracM);
    std::vector<double> Z;
    for(int i=0; i<N; i++) Z.push_back(0);
    
    double dt = 0.1;
    double tNow = 0.5 * dt;
    while(tNow < 1500){
        double w = dt * fDecay->Eval(tNow);
        for(int i=0; i<N; i++){
            double t = X[i] - tNow;
            if(t>-70){
                Z[i] += w * gr0->Eval(t);
            }
        }
        tNow += dt;
    }
    TGraph *gr = new TGraph();
    for(int i=0; i<N; i++){
        gr->SetPoint(i, X[i], Z[i]);
    }
    delete fDecay;
    return gr;
}



void plot_DRS_channel(int event=0, int ch=0, TString filename="./outfile_LG.root")
{
  TreeReader tree(filename);
  tree.get_entry(event);
  
  TGraph *grWF = new TGraph(N_Samples, tree.time().data(), tree.voltages(ch).data());
  grWF->GetXaxis()->SetTitle("t (ns)");
  grWF->GetYaxis()->SetTitle("a (mV)");
  grWF->Draw("APL");
}



void plot_DRS_trigger(int event=0, TString filename="./outfile_LG.root")
{
  TreeReader tree(filename);
  tree.get_entry(event);
  
  TGraph *grWF = new TGraph(N_Samples, tree.time().data(), tree.trigger(0).data());
  grWF->GetXaxis()->SetTitle("t (ns)");
  grWF->GetYaxis()->SetTitle("a (mV)");
  grWF->Draw("APL");
  
  RecoTRG *trg  = new RecoTRG(grWF);
  cout << " Baseline RMS = " << trg->rms() << endl;
  cout << " Amplitude    = " << trg->amp() << endl;
  cout << " Time @ 50%   = " << trg->time() << endl;
  
}



void reco_WF_SDL(int ievt=0, int ch=0)
{
    /*
     * Example of reconstruction for DSB crystal:
     * Run = 200
     * Channels 0, 1, 2, and 3 are in the front
     * Channels 4 and 7 are in the rear
     * Channels 5 and 6 are discarded from the analysis (dead)
     */
    if(!(ch==0 || ch==1 || ch==2 || ch==3 || ch==4 || ch==7)) return;
    TString filename = "./outfile_LG.root";

    
    // SPR for Cerenkov
    TFile *fSPR = new TFile("SPR_SDL.root");
    grCerenkov = (TGraph*)fSPR->Get(Form("grSPR_SDL_ch%d_rear",ch));

    // SPR for Scintillation
    // DSB has two decay components 100ns (13%) and 500ns (87%)
    grScint    = grGetScintillation(grCerenkov, 30,100,500,0.0,0.13);

    // Function for component fit
    TF1 *fFit = new TF1("fFit",funcSDL,-100,800,3);
    fFit->SetLineColor(kOrange+10);
    fFit->SetParName(0,"time jitter");
    fFit->SetParName(1,"Nc x A1pe");
    fFit->SetParName(2,"Ns x A1pe");
  
  
    TreeReader tree(filename);
    tree.get_entry(ievt);

    // Find trigger timing
    TGraph *grTr = new TGraph(N_Samples, tree.time().data(), tree.trigger(0).data());
    RecoTRG *trig = new RecoTRG( grTr );
    delete grTr;
    
    // DRS waveform
    TGraph *grWF = new TGraph(N_Samples, tree.time().data(), tree.voltages(ch).data());

    
    // SDL waveform with 2ns delay, adjusted for trigger timing
    int     N = grWF->GetN();
    double *X = grWF->GetX();
    double *Y = grWF->GetY();
    TGraph *grSDL = new TGraph();
    for(int it=2; it<N; it++){
            grSDL->SetPoint(it-2, X[it] - trig->time(), Y[it-2] - Y[it]);
    }
    fFit->SetParameters(0, 10, 500.);
    fFit->SetParLimits(0, -2, 2);
    fFit->SetParLimits(1, 0, 1000);
    fFit->SetParLimits(2, 0, 100000.);
    grSDL->Fit("fFit","WN","",-60,50);
    
    grSDL->GetXaxis()->SetLimits(-100,50);
    grSDL->GetXaxis()->SetTitle("t - t_{ TRG} (ns)");
    grSDL->GetYaxis()->SetTitle("a(t) - a(t-2ns) (mV)");
          
    TF1 *fC = new TF1("fC",funcSDL,-100,800,3);
    TF1 *fS = new TF1("fS",funcSDL,-100,800,3);
    fC->SetLineColor(kAzure+6);
    fS->SetLineColor(kSpring-1);

    grSDL->Draw("APL");
    fC->SetParameters(fFit->GetParameter(0), fFit->GetParameter(1), 0);
    fS->SetParameters(fFit->GetParameter(0), 0, fFit->GetParameter(2));
    fFit->SetNpx(10000);
    fC->SetNpx(10000);
    fS->SetNpx(10000);
    fFit->SetLineWidth(1);
    fC->SetLineWidth(1);
    fS->SetLineWidth(1);
    
    fFit->Draw("same");
    fC->Draw("same");
    fS->Draw("same");
         
    TLegend *leg = new TLegend(0.54,0.84,0.97,0.99,NULL,"brNDC");
    leg->SetLineColor(1);
    leg->SetLineStyle(1);
    leg->SetLineWidth(1);
    leg->SetFillColor(0);
    leg->SetFillStyle(1001);
    leg->SetTextFont(42);
    leg->SetTextSize(0.033);
    leg->AddEntry(grSDL,"Measurements","pl");
    leg->AddEntry(fFit,"Fit = Cerenkov + Scintillation","l");
    leg->AddEntry(fC,"Cerenkov","l");
    leg->AddEntry(fS,"Scintillation","l");
    leg->Draw(); 
    
}

void generate_WF_from_SDL(double time_jitter = 0, double time_trigger = 155, double NcA1pe = 500, double NsA1pe = 1000, int ch = 3, const std::string& csv_filename = "waveform_data.csv") {
    TFile *fSPR = new TFile("SPR_SDL.root");
    grCerenkov = (TGraph*)fSPR->Get(Form("grSPR_SDL_ch%d_rear", ch));
    if (!grCerenkov) {
        std::cerr << "Error: Cerenkov template not found in SPR_SDL.root for channel " << ch << std::endl;
        return;
    }

    TF1* fitCerenkov = new TF1("fitCerenkov", "[0] * exp([1] * x) + [2]", 100, 600);
    fitCerenkov->SetParameters(1.0, -0.01, 0.001);
    grCerenkov->Fit(fitCerenkov, "Q");

    int n_points_Cerenkov = grCerenkov->GetN();
    double* x_Cerenkov = grCerenkov->GetX();
    double* y_Cerenkov = grCerenkov->GetY();


    std::vector<double> extended_x_Cerenkov, extended_y_Cerenkov;
    for (int i = 0; i < n_points_Cerenkov; ++i) {
      extended_x_Cerenkov.push_back(x_Cerenkov[i]);
      extended_y_Cerenkov.push_back(y_Cerenkov[i]);
    }

    for (double t = 630; t <= 1200; t += 1.0) {
    extended_x_Cerenkov.push_back(t);
    extended_y_Cerenkov.push_back(fitCerenkov->Eval(t)+0.006);
    }

    TGraph* extended_grCerenkov = new TGraph(extended_x_Cerenkov.size(), &extended_x_Cerenkov[0], &extended_y_Cerenkov[0]);
    grScint = grGetScintillation(extended_grCerenkov, 30, 100, 500, 0.0, 0.13);

    // TCanvas *canvas = new TCanvas("canvas", "data", 800, 600);
    // grScint->SetMarkerStyle(20);
    // grScint->SetMarkerColor(kRed);
    // grScint->Draw("AP"); 
    // canvas->SaveAs("fit_comparison.png"); 

    // TCanvas *canvasb = new TCanvas("canvasb", "data", 800, 600);
    // extended_grCerenkov->SetMarkerStyle(20);
    // extended_grCerenkov->SetMarkerColor(kBlue);
    // extended_grCerenkov->Draw("AP"); 
    // canvasb->SaveAs("fit_comparisonb.png"); 



    const int N_Samples_rec = 1024;
    double horizontal_interval = 1.0; 
    std::vector<double> time_points(N_Samples_rec);
    std::vector<double> sdl_points(N_Samples_rec, 0);
    std::vector<double> waveform_points(N_Samples_rec, 0);

    for (int i = 0; i < N_Samples_rec; ++i) {
        time_points[i] = -time_trigger + i * horizontal_interval - time_jitter;
    }



    // for (int i = 0; i < N_Samples_rec; ++i) {
    //     double t = time_points[i];
        
    //     double cerenkov_signal;
    //     if (t < -69) {
    //         cerenkov_signal = 0;
    //     } else {
    //         cerenkov_signal = NcA1pe * grCerenkov->Eval(t);
    //     }
    //     double scint_signal;
    //     if (t < -69) {
    //         scint_signal = 0;
    //     } else {
    //         scint_signal = NsA1pe * grScint->Eval(t);
    //     }
    //     sdl_points[i] = cerenkov_signal + scint_signal;
    // }

        for (int i = 0; i < N_Samples_rec; ++i) {
        double t = time_points[i];
    
        double cerenkov_signal;
        if (t < -69) {
            cerenkov_signal = 0;
        } else {
            cerenkov_signal = NcA1pe * extended_grCerenkov->Eval(t);
        }

        double scint_signal;
        if (t < -69) {
            scint_signal = 0;
        } else {
            scint_signal = NsA1pe * grScint->Eval(t);
        }
        sdl_points[i] = cerenkov_signal + scint_signal;
    }


    for (int i = 0; i < N_Samples_rec; ++i) {
        time_points[i] += time_trigger;
    }

    waveform_points[0] = 400;
    waveform_points[1] = 400;

    for (int i = 2; i < N_Samples_rec; ++i) {
        waveform_points[i] = waveform_points[i - 2] - sdl_points[i];
    }

    std::ofstream csvFile(csv_filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing waveform data." << std::endl;
        return;
    }

    csvFile << "Time(ns),Amplitude(mV)\n";
    for (int i = 0; i < N_Samples_rec; ++i) {
        csvFile << time_points[i] << "," << waveform_points[i] << "\n";
    }
    csvFile.close();

    std::cout << "Waveform data saved to " << csv_filename << std::endl;

    // TFile *outFile = new TFile("generated_waveforms.root", "RECREATE");
    // TTree *tree = new TTree("tree", "Generated Waveform Data");

    // Float_t horizontal_interval_out = horizontal_interval;
    // std::array<std::array<Float_t, N_Samples>, N_Channels> channels = {0};  
    // std::array<std::array<Float_t, N_Samples>, 2> trigger = {0};           

    // for (int i = 0; i < N_Samples; ++i) {
    //     channels[ch][i] = waveform_points[i];
    // }

    // for (int i = 0; i < N_Samples; ++i) {
    //     trigger[0][i] = 0;
    //     trigger[1][i] = 0;  
    // }

    // tree->Branch("horizontal_interval", &horizontal_interval_out, "horizontal_interval/F");
    // tree->Branch("channels", channels.data(), Form("channels[%d][%d]/F", N_Channels, N_Samples));
    // tree->Branch("trigger", trigger.data(), Form("trigger[2][%d]/F", N_Samples));

    // tree->Fill();
    // outFile->Write();
    // outFile->Close();

    // std::cout << "Waveform data saved to generated_waveforms.root" << std::endl;

    TGraph *grWF = new TGraph(N_Samples_rec, &time_points[0], &waveform_points[0]);
    grWF->SetTitle("Reconstructed Original Waveform;Time (ns);Amplitude (mV)");
    grWF->SetLineColor(kBlue);
    grWF->SetLineWidth(2);
    grWF->Draw("AL");

    TLegend *leg = new TLegend(0.6, 0.15, 0.9, 0.3);
    leg->SetBorderSize(0);
    leg->SetFillColor(0);
    leg->AddEntry((TObject*)0, Form("Time Jitter: %.2f ns", time_jitter), "");
    leg->AddEntry((TObject*)0, Form("Nc x A1pe: %.2f", NcA1pe), "");
    leg->AddEntry((TObject*)0, Form("Ns x A1pe: %.2f", NsA1pe), "");
    leg->Draw();

    // grWF->SaveAs("Reconstructed_Waveform.root");
}


void reco_WF_SDL_from_CSV(const std::string &csv_filename, double trigger_time, int ch=0) {
    TFile *fSPR = new TFile("SPR_SDL.root");
    grCerenkov = (TGraph*)fSPR->Get(Form("grSPR_SDL_ch%d_rear", ch));

    grScint = grGetScintillation(grCerenkov, 30, 100, 500, 0.0, 0.13);

    std::vector<double> time_points;
    std::vector<double> voltage_points;

    std::ifstream file(csv_filename);

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string time_str, voltage_str;

        std::getline(ss, time_str, ',');
        std::getline(ss, voltage_str, ',');

        time_points.push_back(std::stod(time_str));
        voltage_points.push_back(std::stod(voltage_str));
    }
    file.close();

    TGraph *grWF = new TGraph(time_points.size(), &time_points[0], &voltage_points[0]);

    int N = grWF->GetN();
    double *X = grWF->GetX();
    double *Y = grWF->GetY();

    TGraph *grSDL = new TGraph();
    for(int it = 2; it < N; ++it) {
        grSDL->SetPoint(it - 2, X[it] - trigger_time, Y[it - 2] - Y[it]);
    }

    TF1 *fFit = new TF1("fFit", funcSDL, -100, 800, 3);
    fFit->SetLineColor(kOrange + 10);
    fFit->SetParName(0, "time jitter");
    fFit->SetParName(1, "Nc x A1pe");
    fFit->SetParName(2, "Ns x A1pe");

    fFit->SetParameters(0, 10, 500.);
    fFit->SetParLimits(0, -2, 2);
    fFit->SetParLimits(1, 0, 1000);
    fFit->SetParLimits(2, 0, 100000.);
    grSDL->Fit("fFit", "WN", "", -60, 50);

    // Plot settings
    grSDL->GetXaxis()->SetLimits(-100, 50);
    grSDL->GetXaxis()->SetTitle("t - t_{TRG} (ns)");
    grSDL->GetYaxis()->SetTitle("a(t) - a(t-2ns) (mV)");

    TF1 *fC = new TF1("fC", funcSDL, -100, 800, 3);
    TF1 *fS = new TF1("fS", funcSDL, -100, 800, 3);
    fC->SetLineColor(kAzure + 6);
    fS->SetLineColor(kSpring - 1);

    grSDL->Draw("APL");
    fC->SetParameters(fFit->GetParameter(0), fFit->GetParameter(1), 0);
    fS->SetParameters(fFit->GetParameter(0), 0, fFit->GetParameter(2));
    fFit->SetNpx(10000);
    fC->SetNpx(10000);
    fS->SetNpx(10000);
    fFit->SetLineWidth(1);
    fC->SetLineWidth(1);
    fS->SetLineWidth(1);

    fFit->Draw("same");
    fC->Draw("same");
    fS->Draw("same");

    TLegend *leg = new TLegend(0.54, 0.84, 0.97, 0.99, NULL, "brNDC");
    leg->SetLineColor(1);
    leg->SetLineStyle(1);
    leg->SetLineWidth(1);
    leg->SetFillColor(0);
    leg->SetFillStyle(1001);
    leg->SetTextFont(42);
    leg->SetTextSize(0.033);
    leg->AddEntry(grSDL, "Measurements", "pl");
    leg->AddEntry(fFit, "Fit = Cerenkov + Scintillation", "l");
    leg->AddEntry(fC, "Cerenkov", "l");
    leg->AddEntry(fS, "Scintillation", "l");
    leg->Draw();
}




// void generate_multiple_waveforms(int n_waveforms, int ch = 3) {
//     TFile *fSPR = new TFile("SPR_SDL.root");
//     grCerenkov = (TGraph*)fSPR->Get(Form("grSPR_SDL_ch%d_rear", ch));

//     TF1* fitCerenkov = new TF1("fitCerenkov", "[0] * exp([1] * x) + [2]", 100, 600);
//     fitCerenkov->SetParameters(1.0, -0.01, 0.001);
//     grCerenkov->Fit(fitCerenkov, "Q");

//     int n_points_Cerenkov = grCerenkov->GetN();
//     double* x_Cerenkov = grCerenkov->GetX();
//     double* y_Cerenkov = grCerenkov->GetY();

//     std::vector<double> extended_x_Cerenkov, extended_y_Cerenkov;
//     for (int i = 0; i < n_points_Cerenkov; ++i) {
//       extended_x_Cerenkov.push_back(x_Cerenkov[i]);
//       extended_y_Cerenkov.push_back(y_Cerenkov[i]);
//     }

//     for (double t = 630; t <= 1200; t += 1.0) {
//         extended_x_Cerenkov.push_back(t);
//         extended_y_Cerenkov.push_back(fitCerenkov->Eval(t) + 0.006);
//     }

//     TGraph* extended_grCerenkov = new TGraph(extended_x_Cerenkov.size(), &extended_x_Cerenkov[0], &extended_y_Cerenkov[0]);
//     grScint = grGetScintillation(extended_grCerenkov, 30, 100, 500, 0.0, 0.13);

//     const int N_Samples_rec = 1024;
//     double horizontal_interval = 1.0;
//     std::vector<double> time_points(N_Samples_rec);
//     std::vector<double> sdl_points(N_Samples_rec, 0);
//     std::vector<double> waveform_points(N_Samples_rec, 0);

//     std::ofstream csvFile("waveform_data_10000.csv");
//     if (!csvFile.is_open()) {
//         std::cerr << "Error: Unable to open CSV file for writing." << std::endl;
//         return;
//     }

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> dis_time_jitter(-0.4, 0.4);
//     std::uniform_real_distribution<> dis_time_trigger(152.0, 158.0);
//     std::uniform_real_distribution<> dis_NcA1pe(0.0, 250.0);
//     std::uniform_real_distribution<> dis_NsA1pe(0.0, 6300.0);

//     for (int n = 0; n < n_waveforms; ++n) {
//         double time_jitter = dis_time_jitter(gen);
//         double time_trigger = dis_time_trigger(gen);
//         double NcA1pe = dis_NcA1pe(gen);
//         double NsA1pe = dis_NsA1pe(gen);

//         for (int i = 0; i < N_Samples_rec; ++i) {
//             time_points[i] = -time_trigger + i * horizontal_interval - time_jitter;
//         }

//         for (int i = 0; i < N_Samples_rec; ++i) {
//             double t = time_points[i];

//             double cerenkov_signal;
//             if (t < -69) {
//                 cerenkov_signal = 0;
//             } else {
//                 cerenkov_signal = NcA1pe * extended_grCerenkov->Eval(t);
//             }

//             double scint_signal;
//             if (t < -69) {
//                 scint_signal = 0;
//             } else {
//                 scint_signal = NsA1pe * grScint->Eval(t);
//             }
//             sdl_points[i] = cerenkov_signal + scint_signal;
//         }

//         waveform_points[0] = 400;
//         waveform_points[1] = 400;
//         for (int i = 2; i < N_Samples_rec; ++i) {
//             waveform_points[i] = waveform_points[i - 2] - sdl_points[i];
//         }

//         // Write the waveform data for this event as a new column in the CSV
//         for (int i = 0; i < N_Samples_rec; ++i) {
//             csvFile << waveform_points[i] << ",";
//         }
//         csvFile << NcA1pe << "," << NsA1pe << "\n";
//     }

//     csvFile.close();
//     std::cout << "Waveform data saved to waveform_data_10000.csv" << std::endl;
// }

void generate_multiple_waveforms(int n_waveforms, int ch = 3, double NcA1pe_min = 0.0, double NcA1pe_max = 250.0, double NsA1pe_min = 0.0, double NsA1pe_max = 6300.0, const std::string& csv_filename = "waveform_data.csv") {
    TFile *fSPR = new TFile("SPR_SDL.root");
    TGraph* grCerenkov = (TGraph*)fSPR->Get(Form("grSPR_SDL_ch%d_rear", ch));

    TF1* fitCerenkov = new TF1("fitCerenkov", "[0] * exp([1] * x) + [2]", 100, 600);
    fitCerenkov->SetParameters(1.0, -0.01, 0.001);
    grCerenkov->Fit(fitCerenkov, "Q");

    int n_points_Cerenkov = grCerenkov->GetN();
    double* x_Cerenkov = grCerenkov->GetX();
    double* y_Cerenkov = grCerenkov->GetY();

    std::vector<double> extended_x_Cerenkov, extended_y_Cerenkov;
    for (int i = 0; i < n_points_Cerenkov; ++i) {
        extended_x_Cerenkov.push_back(x_Cerenkov[i]);
        extended_y_Cerenkov.push_back(y_Cerenkov[i]);
    }

    for (double t = 630; t <= 1200; t += 1.0) {
        extended_x_Cerenkov.push_back(t);
        extended_y_Cerenkov.push_back(fitCerenkov->Eval(t) + 0.006);
    }

    TGraph* extended_grCerenkov = new TGraph(extended_x_Cerenkov.size(), &extended_x_Cerenkov[0], &extended_y_Cerenkov[0]);
    TGraph* grScint = grGetScintillation(extended_grCerenkov, 30, 100, 500, 0.0, 0.13);

    const int N_Samples_rec = 1024;
    double horizontal_interval = 1.0;
    std::vector<double> time_points(N_Samples_rec);
    std::vector<double> sdl_points(N_Samples_rec, 0);
    std::vector<double> waveform_points(N_Samples_rec, 0);

    std::ofstream csvFile(csv_filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Unable to open CSV file for writing." << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_time_jitter(-0.4, 0.4);
    std::uniform_real_distribution<> dis_time_trigger(152.0, 158.0);
    std::uniform_real_distribution<> dis_NcA1pe(NcA1pe_min, NcA1pe_max);
    std::uniform_real_distribution<> dis_NsA1pe(NsA1pe_min, NsA1pe_max);
    
    // Main distribution range
    std::uniform_real_distribution<> dis_waveform_initial_main(395.0, 415.0);
    // Auxiliary distribution range
    std::uniform_real_distribution<> dis_waveform_initial_aux(300.0, 440.0);

    // Probability to select the main distribution
    std::uniform_real_distribution<> dis_probability(0.0, 1.0);
    double main_distribution_prob = 0.9; // 90% chance to select the main distribution

    for (int n = 0; n < n_waveforms; ++n) {
        double time_jitter = dis_time_jitter(gen);
        double time_trigger = dis_time_trigger(gen);
        double NcA1pe = dis_NcA1pe(gen);
        double NsA1pe = dis_NsA1pe(gen);

        for (int i = 0; i < N_Samples_rec; ++i) {
            time_points[i] = -time_trigger + i * horizontal_interval - time_jitter;
        }

        for (int i = 0; i < N_Samples_rec; ++i) {
            double t = time_points[i];

            double cerenkov_signal;
            if (t < -69) {
                cerenkov_signal = 0;
            } else {
                cerenkov_signal = NcA1pe * extended_grCerenkov->Eval(t);
            }

            double scint_signal;
            if (t < -69) {
                scint_signal = 0;
            } else {
                scint_signal = NsA1pe * grScint->Eval(t);
            }
            sdl_points[i] = cerenkov_signal + scint_signal;
        }

        // Generate random initial values for the waveform using mixed distribution
        waveform_points[0] = (dis_probability(gen) < main_distribution_prob) ? dis_waveform_initial_main(gen) : dis_waveform_initial_aux(gen);
        waveform_points[1] = (dis_probability(gen) < main_distribution_prob) ? dis_waveform_initial_main(gen) : dis_waveform_initial_aux(gen);

        for (int i = 2; i < N_Samples_rec; ++i) {
            waveform_points[i] = waveform_points[i - 2] - sdl_points[i];
        }

        // Write the waveform data for this event as a new column in the CSV
        for (int i = 0; i < N_Samples_rec; ++i) {
            csvFile << waveform_points[i] << ",";
        }
        csvFile << NcA1pe << "," << NsA1pe << "\n";
    }

    csvFile.close();
    std::cout << "Waveform data saved to " << csv_filename << std::endl;
}
