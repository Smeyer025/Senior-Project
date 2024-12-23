/**
 * App Component: Overall body of the website
 */

import Header from "./components/Header/Header";
import AnalysisContainer from "./components/AnalysisContainer/AnalysisContainer";
import MenuContainer from "./components/MenuContainer/MenuContainer";

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
    <html>
      <body>
      <Header>
        Sentiment Analysis
      </Header>
      <AnalysisContainer />
      <MenuContainer />
      </body>
    </html>
  );
}

export default App;