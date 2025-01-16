#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

const double ns = 1.0;
const double coarse = 8.0*ns;
const double TIME_WINDOW = 450*ns; 
const int THRESH = 400;

double time_base = -1;

void loadCSVData(
    const std::string& filename,
    std::vector<double>& column1,
    std::vector<int>& column2,
    std::vector<int>& column3,
    std::vector<int>& column4
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open file " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string cell;

        double value1; 
        int value2, value3, value4;
        if (std::getline(lineStream, cell, ',')) value1 = std::stod(cell);
        else throw std::runtime_error("Error: Invalid data in column 1");
        if (std::getline(lineStream, cell, ',')) value2 = std::stod(cell);
        else throw std::runtime_error("Error: Invalid data in column 2");
        if (std::getline(lineStream, cell, ',')) value3 = std::stod(cell);
        else throw std::runtime_error("Error: Invalid data in column 3");
        if (std::getline(lineStream, cell, ',')) value4 = std::stod(cell);
        else throw std::runtime_error("Error: Invalid data in column 4");

        column1.push_back(value1);
        column2.push_back(value2);
        column3.push_back(value3);
        column4.push_back(value4);
    }

    file.close();
}

int main(){
    std::vector<double> times;
    std::vector<int> charges;
    std::vector<int> slot;
    std::vector<int> channel; 
    std::vector<int> event_id;

    double event_start_time = 0;

    // load in the text file 
    std::cout<<"Loading Data"<<std::endl;
    loadCSVData("hits.csv", times, charges, slot, channel);
    std::cout <<"Loaded "<<times.size()<<" hits"<<std::endl;
    std::vector<int> to_assign;

    bool in_event =false;
    int event_counter = 0;

    for(int index=0; index<times.size(); index++){
        event_id.push_back(-1);
    }

    int min_index=0;
    int max_index=1;
    int n_in_range = 0;

    // first, we slide the max index over until the time is greater than the time window 
    while((times[max_index] - times[min_index])<TIME_WINDOW){
        if(max_index>times.size()-1){
            max_index -= 1; // shift it back over so that max index is in the window
            break; 
        }else{
            max_index++;
        }
    }

    n_in_range = max_index - min_index+1; 

    // if there are already enough hits then we assign all of these to the first event 
    if(n_in_range>THRESH){
        // the max
        for(int index=0; index<max_index; index++){
            event_id[index] = event_counter;
        }
        in_event = true;
    }
    uint index = 0;
    while(index<times.size()){

        // update the lower and upper bounds 

        // while the min index is low enough that we look at a non-valid one, step it up!
        while((times[index] - times[min_index])>TIME_WINDOW){
            if(index==min_index){
                break;
            }
            if(min_index>times.size()-1){
                break; 
            }else{
                min_index++;
            }
        }
        // min_index is now the earliest hit that is still in the window 

        // as we scan to the right, any we add while already in an event need to be added to the current event 
        while((times[max_index] - times[index])<TIME_WINDOW){
            if(max_index>times.size()-1){
                break; 
            }else{
                // if we're in an event then we will need to assign this new index to the currently considered event 
                if(in_event){
                    to_assign.push_back(max_index);
                }
                // after that, we step the index forward! 
                max_index++;
                
            }
        }
        // max index is large enough to be invalid, so let's shift it over 
        max_index -=1;

        if(min_index>index){
            min_index=index;
        }
        if(max_index<index){
            max_index=index;
        }
        
        // are there enough hits to call this an event. 
        /*
            We check the number of hits around this one
            If we're above the threshold,
                we check if we're already in an event 
                    And if so, then we add the newly scanned hits to this one
                    otherwise, we have to iterate the event counter and assign all of the hits 

            Otherwise, we signal the exit of an event

        */
        n_in_range = max_index - min_index+1;
        if (max_index>index){
            index = max_index;
        }else{
            index ++;
        }
        if (n_in_range>THRESH){
            if(in_event){
                // only assign the newest values 
                for(int i=0; i<to_assign.size();i++){
                    event_id[to_assign[i]] = event_counter;
                }
            }else{
                
                in_event = true;
                event_counter++;
                std::cout<<"Found event " << event_counter<<std::endl;
                for(int i=min_index; i<=max_index; i++){
                    event_id[i] = event_counter;
                }
            }
        }else{
            if (in_event){
                // we may have just exited an event... so we might need to assign things 
                for(int i=0; i<to_assign.size();i++){
                    event_id[to_assign[i]] = event_counter;
                }
            }
            in_event = false;
        }
        
        to_assign.clear();
    }

    std::cout <<"Saving events file"<<std::endl;
    std::ofstream file("./events.csv");
    file << "Hit no, time, charge, slot, channel, event" <<std::endl;;
    std::string line;
    int nhits = 0;
    int last_event_id = -1;

    for(int i=0; i<times.size(); i++){
        if(event_id[i]==-1){
            continue;
        }else{
            if(last_event_id!=event_id[i]){
                time_base = times[i];
                }
            last_event_id = event_id[i];

            nhits++;
            file<<i;
            file<<", ";
            file<<times[i]-time_base;
            file<<", ";
            file<<charges[i];
            file<<", ";
            file<<slot[i];
            file<<", ";
            file<<channel[i];
            file<<", ";
            file<<event_id[i];
            file<<std::endl;
        }
    }

    std::cout<<"counted "<<nhits<<" hits in "<<last_event_id<<" events"<<std::endl;

    return 0;
}