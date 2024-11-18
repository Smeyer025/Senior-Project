/**
 * App Component: Overall body of the website
 */

import Header from "./components/Header/Header";
import AnalysisContainer from "./components/AnalysisContainer/AnalysisContainer";
import OutputContainer from "./components/OutputContainer/OutputContainer";

/**
 * App()
 * 
 * NAME
 *    App() - Handles overall webpage layout
 * 
 * SYNOPSIS
 *    React.JSX.Element App()
 * 
 * DESCRIPTION
 *    This function exports the overall webpage layout and 
 *    allows it to be called from the index.js file 
 */
function App() {
  return (
    <>
      <Header>
        Sentiment Analysis
      </Header>
      <AnalysisContainer />
    </>
  );
}

export default App;