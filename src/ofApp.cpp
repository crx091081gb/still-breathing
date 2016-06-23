#include "ofApp.h"
using namespace cv;
using namespace ofxCv;

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetLogLevel(OF_LOG_VERBOSE);
    
    //gui stuff
    gui.setup();
    gui.add(alpha.setup("alpha", 15, 0, 20));
    gui.add(cutoff.setup("cutoff", 16, 0, 20));
    gui.add(min_freq.setup("min_freq", 0.05, 0, 2));
    gui.add(max_freq.setup("max_freq", 0.4, 0, 2));
    gui.add(chrome.setup("chrome", 0.1, 0, 2));
    gui.add(breathBlobThreshold.set("Breath Blob Threshold", 220, 0, 255));
    gui.add(minArea.set("Min area", 4, 0, 30));
    gui.add(maxArea.set("Max area", 10, 1, 60));
    gui.add(invert.set("Invert", false));
    gui.add(depthThreshold.set("Depth threshold", 175, 0, 255));
    gui.add(minDepthArea.set("Min people blob size", 100, 1, 500));
    gui.add(maxDepthArea.set("Max people blob size", 200, 1, 1000));
    gui.add(slitRatio.set("Slit Position", 0.75, 0.0, 1.0));
    gui.add(trackerPersistence.set("Tracker Persistence", 30, 0, 100));
    gui.add(trackerMaxDistance.set("Tracker Max Distance", 30, 0, 100));
    
    
    // enable depth->video image calibration
    kinect.setRegistration(true);
    
    //set up kinect
    kinect.init();
    kinect.open();
    
    if(kinect.isConnected()) {
        ofLogNotice() << "sensor-emitter dist: " << kinect.getSensorEmitterDistance() << "cm";
        ofLogNotice() << "sensor-camera dist:  " << kinect.getSensorCameraDistance() << "cm";
        ofLogNotice() << "zero plane pixel size: " << kinect.getZeroPlanePixelSize() << "mm";
        ofLogNotice() << "zero plane dist: " << kinect.getZeroPlaneDistance() << "mm";
        
        //set use live kinect true
        gui.add(live.set("Live Kinect", true));
        
    } else {
        //set use live kinect false
        gui.add(live.set("Live Kinect", false));
    }
    
    //allocate textures
    ofColourImg.allocate(kinect.width, kinect.height, OF_IMAGE_COLOR);
    ofDepthImg.allocate(kinect.width, kinect.height, OF_IMAGE_COLOR);
    ofThreshImg.allocate(kinect.width, kinect.height, OF_IMAGE_COLOR);
    
    //TODO ADD IFNDEF AND PUT WEBCAM / KINECT VERSIONS IN ONE FILE
    //video.initGrabber(640, 480);
    
    
    //set up image buffers and contour finders
    for (int i=0; i<4; i++) {
        //separate images and contourfinders necessary as each slitscan
        //is taken at different height of original colour image
        ofImage img;
        img.load("cat.jpg");
        img.resize(640, 480);
        scanSlices.push_back(img);
        
        ofxCv::ContourFinder cf;
        cf.setMinAreaRadius(minArea);
        cf.setMaxAreaRadius(maxArea);
        cf.setThreshold(breathBlobThreshold);
        cf.findContours(img);
        contourFinders.push_back(cf);
        
    }
    
    //track blobs in thresholded depth map and assume their humans
    tracker.setPersistence(trackerPersistence);
    tracker.setMaximumDistance(trackerMaxDistance);
    
    //video recorder settings for kinect signal
    fileName1 = "kinect-video";
    fileName2 = "kinect-depth";
    fileExt = ".mov";
    
    vidRecorder1.setVideoCodec("mpeg4");
    vidRecorder1.setVideoBitrate("800k");
    
    vidRecorder2.setVideoCodec("mpeg4");
    vidRecorder2.setVideoBitrate("800k");
    
    ofAddListener(vidRecorder1.outputFileCompleteEvent, this, &ofApp::recordingComplete);
    ofAddListener(vidRecorder2.outputFileCompleteEvent, this, &ofApp::recordingComplete);
    
    recordedVideoPlayback.load("kinect-video-test.mov");
    recordedDepthPlayback.load("kinect-depth-test.mov");
    recordedVideoPlayback.play();
    recordedDepthPlayback.play();
   
    //of screen stuff
    ofSetFrameRate(30);
    ofSetVerticalSync(true);
    ofBackground(0);
    ofEnableAlphaBlending();
}

//--------------------------------------------------------------
void ofApp::update(){
    updateParams();
    ofBackground(100, 100, 100);
    
    cv::Mat colourImg;
    cv::Mat depthImg;
    
    if (live) {
        if (kinect.isConnected()) {
            kinect.update();
            // there is a new frame and we are connected
            if(kinect.isFrameNew()) {
                //read in colour image
                colourImg = cv::Mat(kinect.height, kinect.width, CV_8UC3, kinect.getPixels(), 0);
                
                //convert to of format for display and recording
                toOf(colourImg, ofColourImg);
                ofColourImg.update();
                
                //record colour image
                if (bRecording) {
                    bool success = vidRecorder1.addFrame(kinect.getPixels());
                    if (!success) {
                        ofLogWarning("This frame was not added");
                    }
                }
                
                // Check if the video recorder encountered any error while writing video frame or audio smaples.
                if (vidRecorder1.hasVideoError()) {
                    ofLogWarning("The video recorder failed to write some frames!");
                }
                
                if (vidRecorder1.hasAudioError()) {
                    ofLogWarning("The video recorder failed to write some audio samples!");
                }
                
                //read in depth image
                depthImg = cv::Mat(kinect.height, kinect.width, CV_8UC1, kinect.getDepthPixels(), 0);
                
                //force 3 channel copy for display and recording
                cv::Mat bgr;
                cv::cvtColor(depthImg, bgr, CV_GRAY2BGR);
                
                //convert to of format for display and recording
                toOf(bgr, ofDepthImg);
                ofDepthImg.update();
                
                //record depth image
                if (bRecording) {
                    bool success = vidRecorder2.addFrame(ofDepthImg.getPixels());
                    if (!success) {
                        ofLogWarning("This frame was not added");
                    }
                }
                
                //recieved a new frames worth of data
                newFrame = true;
            } else {
                //nothing happened so don't try and process data later
                newFrame = false;
            }
        } else {
            //lost connection
            live = false;
            newFrame = false;
        }
    } else {
        //playing back test video footage
        recordedVideoPlayback.update();
        recordedDepthPlayback.update();
        if (recordedVideoPlayback.isFrameNew() && recordedDepthPlayback.isFrameNew()) {
            //read in colour image
            colourImg = cv::Mat(recordedVideoPlayback.getHeight(), recordedVideoPlayback.getWidth(), CV_8UC3, recordedVideoPlayback.getPixels(), 0);
            
            //convert to of format for display and recording
            toOf(colourImg, ofColourImg);
            ofColourImg.update();
            
            //read in depth image
            cv::Mat bgr = cv::Mat(recordedDepthPlayback.getHeight(), recordedDepthPlayback.getWidth(), CV_8UC3, recordedDepthPlayback.getPixels(), 0);
            //force back to 1 channel
            cv::cvtColor(bgr, depthImg, CV_BGR2GRAY);
            toOf(bgr, ofDepthImg);
            ofDepthImg.update();
            
            newFrame = true;
        } else {
            newFrame = false;
        }
    }
    
    if (newFrame) {
        
        //feed colour image to evm filter
        evm.update(colourImg);
        
        //background cull
        threshold(depthImg, depthThreshold);
        toOf(depthImg, ofThreshImg);
        ofThreshImg.update();
        
        //blob track people
        depthContourFinder.findContours(depthImg);
        tracker.track(depthContourFinder.getBoundingRects());
        
        
        
        //iterate thru blobs
        int i=0;
        for (auto &follower : tracker.getFollowers() ) {
            //try to work out chest position by assuming it lies a certain percentage way up the bounding rectangle
            ofVec2f b;
            b.y = follower.bottom + (slitRatio*follower.height);
            b.x = follower.smooth.x;
            
            
            //TODO get the age and velocity
            //cout << follower.getLabel() << " age " << tracker.getAge(follower.getLabel()) << endl;

            //get the four oldest (this code is incorrect)
            //and slitscan at chest pos
            if (i < 4) {
                scanSlice(evm.result.getPixels(), scanSlices[i].getPixels(), b.y);
                scanSlices[i].update();
                contourFinders[i].findContours(scanSlices[i]);
            }
            i++;
        }
    }
    
    //fps in window title
    std::stringstream strm;
    strm << "fps: " << ofGetFrameRate();
    ofSetWindowTitle(strm.str());
}

//--------------------------------------------------------------
void ofApp::draw(){
    //draw camera feed
    ofSetColor(255,255,255);
    video.draw(0, 0);
    ofColourImg.draw(0,0);
    
    //draw evm
    ofPushMatrix();
    ofTranslate(kinect.getWidth(), 0);
    evm.draw();
    ofPopMatrix();
    
    //draw depth map
    ofDepthImg.draw(kinect.getWidth(), kinect.getHeight());
    ofSetColor(255,0,0,128);
    ofThreshImg.draw(kinect.width, kinect.height);
    ofSetColor(255,255,255,255);
    ofPushMatrix();
    ofTranslate(kinect.getWidth(), kinect.getHeight());
    depthContourFinder.draw();
    ofPopMatrix();
    //ofSort(tracker.getFollowers());
    
    //draw blob tracked humans
    int i = 0;
    for (BlobPeople &follower : tracker.getFollowers() ) {

        follower.draw();
        //draw chest balls
        ofVec2f b;
        b.y = follower.bottom + (slitRatio*follower.height);
        b.x = follower.smooth.x;
        ofSetColor(255,255,255);
        ofPushMatrix();
        ofTranslate(kinect.getWidth(), kinect.getHeight());
        ofDrawCircle(b, 10);
        ofPopMatrix();
        //draw image buffers and contour finders
        if (i < 4) {
            //void ofImage_::drawSubsection(float x, float y, float w, float h, float sx, float sy)
            int x = follower.left;
            int y = 0;
            int w = follower.right - x;
            int h = kinect.getHeight();
            ofPushMatrix();
            //ofPushStyle();
            ofTranslate(0,kinect.getHeight());
            ofSetColor(255);
            scanSlices[i].drawSubsection(x, y, w, h, x, y);
            ofSetColor(0);
            ofSetLineWidth(3);
            contourFinders[i].draw();
            //ofPopStyle();
            ofPopMatrix();
        }
        i++;
    }
    
    //draw gui
    gui.draw();
}

//--------------------------------------------------------------
void ofApp::exit() {
    ofRemoveListener(vidRecorder1.outputFileCompleteEvent, this, &ofApp::recordingComplete);
    ofRemoveListener(vidRecorder2.outputFileCompleteEvent, this, &ofApp::recordingComplete);
    vidRecorder1.close();
    vidRecorder2.close();
    kinect.close();
}
//--------------------------------------------------------------
void ofApp::updateParams()
{
    //update evm and cv parameters
    evm.amplification(alpha);
    evm.cutoff(cutoff);
    evm.freqMin(min_freq);
    evm.freqMax(max_freq);
    evm.chromeAttenuation(chrome);
    
    depthContourFinder.setMinAreaRadius(minDepthArea);
    depthContourFinder.setMaxAreaRadius(maxDepthArea);
    
    tracker.setPersistence(trackerPersistence);
    tracker.setMaximumDistance(trackerMaxDistance);
    
    for (auto &cf : contourFinders) {
        cf.setMinAreaRadius(minArea);
        cf.setMaxAreaRadius(maxArea);
        cf.setThreshold(breathBlobThreshold);
    }
}
//--------------------------------------------------------------
void ofApp::scanSlice(ofPixels_<float>& src, ofPixels& dst, int _offset) {

    size_t bytesPerLine = (dst.getWidth() * dst.getBytesPerPixel());
    size_t offset = bytesPerLine * _offset;
    
    for (size_t i = 0; i < (dst.getWidth()-1)*dst.getBytesPerPixel(); i+=dst.getBytesPerPixel()) {
        dst[i + 0] = ofClamp(255*src[offset + i + 0], 0, 255);
        dst[i + 1] = ofClamp(255*src[offset + i + 1], 0, 255);
        dst[i + 2] = ofClamp(255*src[offset + i + 2], 0, 255);
    }
    
    for(int l = dst.getHeight()-1; l >= 1; l--) {
        memcpy(&dst[l*bytesPerLine], &dst[(l-1)*bytesPerLine], bytesPerLine);
    }
    
}
//--------------------------------------------------------------
void ofApp::recordingComplete(ofxVideoRecorderOutputFileCompleteEventArgs& args){
    cout << "The recorded video file is now complete." << endl;
}
//--------------------------------------------------------------
bool ofApp::sortPeople(BlobPeople &b1, BlobPeople &b2) {

}
//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    if(key=='r'){
        bRecording = !bRecording;
        if(bRecording && !vidRecorder1.isInitialized() && !vidRecorder2.isInitialized()) {
            vidRecorder1.setup(fileName1+ofGetTimestampString()+fileExt, kinect.getWidth(), kinect.getHeight(), 30);
            vidRecorder2.setup(fileName2+ofGetTimestampString()+fileExt, kinect.getWidth(), kinect.getHeight(), 30);
            //          vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, vidGrabber.getWidth(), vidGrabber.getHeight(), 30); // no audio
            //            vidRecorder.setup(fileName+ofGetTimestampString()+fileExt, 0,0,0, sampleRate, channels); // no video
            //          vidRecorder.setupCustomOutput(vidGrabber.getWidth(), vidGrabber.getHeight(), 30, sampleRate, channels, "-vcodec mpeg4 -b 1600k -acodec mp2 -ab 128k -f mpegts udp://localhost:1234"); // for custom ffmpeg output string (streaming, etc)
            
            // Start recording
            vidRecorder1.start();
            vidRecorder2.start();
        }
        else if(!bRecording && vidRecorder1.isInitialized() && vidRecorder2.isInitialized()) {
            vidRecorder1.setPaused(true);
            vidRecorder2.setPaused(true);
        }
        else if(bRecording && vidRecorder1.isInitialized() && vidRecorder2.isInitialized()) {
            vidRecorder1.setPaused(false);
            vidRecorder2.setPaused(false);
        }
    }
    if(key=='c'){
        bRecording = false;
        vidRecorder1.close();
        vidRecorder2.close();
    }
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

void BlobPeople::setup(const cv::Rect& track) {
    color.setHsb(ofRandom(0, 255), 255, 255);
    cur = toOf(track).getCenter();
    smooth = cur;
}

void BlobPeople::update(const cv::Rect& track) {
    cur = toOf(track).getCenter();
    smooth.interpolate(cur, 0.5);
    height = track.tl().y-track.br().y;
    bottom = track.br().y;
    left = track.tl().x;
    right = track.br().x;
    
    
}

void BlobPeople::draw() {
    ofPushStyle();
    ofSetColor(color);
    ofDrawBitmapString(ofToString(label),cur);
    ofDrawCircle(smooth, 10);
    ofPopStyle();
}
