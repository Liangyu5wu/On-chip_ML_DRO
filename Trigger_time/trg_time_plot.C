#include <TFile.h>
#include <TGraph.h>
#include <iostream>

void trg_time_plot() {
    TString filename = "./outfile_LG.root";
    const int N = 32501;

    std::vector<double> eventIDs;
    std::vector<double> trgTimes;
    std::vector<double> badeventIDs;
    double badnum = 0;
    TreeReader tree(filename);

    TH1D *hTriggerTime = new TH1D("hTriggerTime", "Trigger Time Distribution;Trigger Time (ns);Counts", 100, 140, 170);

    for (int evt = 0; evt < N; evt++) {
        tree.get_entry(evt);
        
        TGraph *grWF = new TGraph(N_Samples, tree.time().data(), tree.trigger(0).data());
        RecoTRG *trg  = new RecoTRG(grWF);

        double trg_time = trg->time();
        
        eventIDs.push_back(evt);
        trgTimes.push_back(trg_time);
        hTriggerTime->Fill(trg_time);

        if (trg_time < 145) {
            badnum++;
            badeventIDs.push_back(evt);
        }

        if (evt % 1000 == 0) {
            std::cout << "Processed event: " << evt << std::endl;
            gSystem->ProcessEvents();
        }

        delete grWF;
        delete trg;
    }

    std::cout << "The count of bad events: " << badnum << std::endl;
    std::cout << "Bad Event IDs: ";
    for (size_t i = 0; i < badeventIDs.size(); i++) {
    std::cout << badeventIDs[i] << " ";
    }
    std::cout << std::endl;

    TGraph *graph = new TGraph(N, eventIDs.data(), trgTimes.data());
     graph->SetTitle("Trigger Time vs Event;Event ID;Trigger Time (ns)");
    
    TFile *outFile = new TFile("trg_time_output.root", "RECREATE");
    graph->Write("trg_time_graph");
    hTriggerTime->Write();
    outFile->Close();

    std::cout << "Saved trg_time_output.root with TGraph." << std::endl;
}
